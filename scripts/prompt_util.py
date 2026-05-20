import numpy as np
import pandas as pd
import re
import json
from itertools import repeat, batched
from collections.abc import Iterator

_LONG_SOLVE_IGNORE_PREFIX = 'English word: '
_SHORT_SOLVE_IGNORE_PREFIX = 'English: '
_TRANSLATE_SEP_TOKEN = '/ '
_TRANSLATE_IGNORE_REGEX = r'^.*/ *'
_IGNORE_AFTER_NEWLINE = r'\n.*$'

SOLVE_PROMPTS = {'solve', 'long-solve', 'short-solve', 'terse-solve'}
SHORT_SOLVE_PROMPTS = {'short-solve', 'terse-solve'}
_IGNORED_PREFIXES = {
    p: _SHORT_SOLVE_IGNORE_PREFIX if (p in SHORT_SOLVE_PROMPTS) else
    _LONG_SOLVE_IGNORE_PREFIX
    for p in SOLVE_PROMPTS
    }

TRANSLATE_PROMPT = {'translate'}
ONE_ZERO_PROMPTS = {'calque', 'transliteration', 'lexical_ambiguity'}


def clean_prompting_output(output: pd.Series, prompt: str) -> pd.Series:
    if prompt in ONE_ZERO_PROMPTS:
        # No need to clean really (at least with GPT-5)
        return output.astype(str).str.strip()
    if prompt in SOLVE_PROMPTS:
        output = output.str.removeprefix(_IGNORED_PREFIXES[prompt]).str.lower()
    else:
        assert prompt == TRANSLATE_PROMPT
        output = output.str.replace(
            _TRANSLATE_IGNORE_REGEX, '', regex=True,
            flags=re.DOTALL  # important, may be multi-line!
            )
    return output.str.replace(
        _IGNORE_AFTER_NEWLINE, '', regex=True,
        flags=re.DOTALL  # important, may be multi-line!
        ).str.lower()


DEFAULT_SCALE_TOKENS = ['1', '2', '3', '4', '5']
DEFAULT_SCALE_VALUES = np.linspace(0, 1, 5)


def _log_probs2num(
    token_logprobs: list,
    tokens: list[str] = DEFAULT_SCALE_TOKENS,
    values: list[float] | np.ndarray = DEFAULT_SCALE_VALUES,
    temperature: float | None = None
    ) -> float:
    '''
    G-Eval-style probability-weighting
    '''
    if token_logprobs:
        tlp = token_logprobs[0]
        t2p = dict(zip(tlp['top_tokens'], tlp['top_logprobs']))
        lps = [t2p.get(t, np.nan) for t in tokens]
        if temperature is not None:
            lps = np.array(lps) / temperature
        weights = np.nan_to_num(
            np.exp(lps),
            nan=0.0  # assign weight 0 if not present in top-k
            )
        if not weights.any():
            weights = np.ones_like(weights)   # equal weight if all weights 0
    else:
        weights = np.ones_like(values)

    return np.average(values, weights=weights)

def _log_probs2nums_iter(
    token_logprobs: list,
    tokens: list[str] = DEFAULT_SCALE_TOKENS,
    values: list[float] | np.ndarray = DEFAULT_SCALE_VALUES,
    temperature: float | None = None,
    sep: str = ',',
    length: int | None = None
    ) -> Iterator[float]:

    # TODO proper exceptions
    n_tokens = len(token_logprobs)
#     assert n_tokens % 2 == 1, (
#         token_logprobs,
#         [tlp['token'] for tlp in token_logprobs]
#         )
    assert length is None or ((n_tokens + 1) // 2 == length)

    for tlp, *sep_tlp in batched(token_logprobs, 2):
        assert (
            not sep_tlp or
            sep_tlp[0]['token'] == sep
            or sep_tlp[0]['token'] in IGNORED_SPECIAL_TOKENS    # TODO: if last
            )
        yield _log_probs2num([tlp], tokens, values, temperature)

def _log_probs2num_list(
    token_logprobs: list,
    tokens: list[str] = DEFAULT_SCALE_TOKENS,
    values: list[float] | np.ndarray = DEFAULT_SCALE_VALUES,
    temperature: float | None = None,
    sep: str = ',',
    length: int | None = None
    ) -> list[float]:
    '''
    Like _log_probs2num, but for a comma separated list.
    '''
    return list(_log_probs2nums_iter(
        token_logprobs, tokens, values, temperature, sep,  length
        ))

def prompting_logprobs2num(
    logprobs: pd.Series,
    tokens: list[str] = DEFAULT_SCALE_TOKENS,
    values: list[float] | np.ndarray = DEFAULT_SCALE_VALUES,
    temperature: float | None = None
    ) -> pd.Series:
    '''
    Apply G-Eval-style probability-weighting
    '''
    return logprobs.apply(lambda x: _log_probs2num(x, tokens, values, temperature))


def prompting_logprobs2nums_split(
    logprobs: pd.Series,
    tokens: list[str] = DEFAULT_SCALE_TOKENS,
    values: list[float] | np.ndarray = DEFAULT_SCALE_VALUES,
    temperature: float | None = None,
    columns: list | None = None
    ) -> pd.Series:
    list_len = len(columns) if (columns is not None) else None
    num_lists = logprobs.apply(
        lambda x: _log_probs2num_list(
            x, tokens, values, temperature, length=list_len
            )
        )
    df = pd.DataFrame(num_lists.tolist(), index=num_lists.index)
    assert not df.isna().any().any()
    if columns is not None:
        assert len(df.columns) == len(columns)
        df.columns = columns

    return df


def _log_prob2prob(logprob: float, t: float | None):
    # temperature scaling for a single log-probability
    if t is None or logprob == 0:
        return np.exp(logprob)

    p   = np.exp(logprob / t)
    pc  = np.exp(np.log(1 - np.exp(logprob)) / t)  # complement

    return p / (p + pc)

IGNORED_SPECIAL_TOKENS = {
    '<｜end▁of▁sentence｜>'   # Used by DeepSeek (sometimes!?)
    }

def _log_probs2prob(
    token_logprobs: list,
    target: str,
    ignore_prefix: str = None,
    ignore_up_to_token: str = None,  # last occurrence if there are multiple
    temperature: float | None = None
    ) -> float:
    '''
    Estimate probability of decoding `target` from `token_logprobs`, while ignoring
    `ignore_prefix` if present.
    '''
    # We lowercase tokens, and expect other params to be already lowercased:
    assert target == target.lower()
    assert ignore_prefix is None or (ignore_prefix == ignore_prefix.lower())
    assert ignore_up_to_token is None or (
        ignore_up_to_token == ignore_up_to_token.lower()
        )
    assert (ignore_prefix is None) or (ignore_up_to_token is None)

    decoded = ''
    decoded_lps = []

    token_logprobs = [
        tlp for tlp in token_logprobs
        if tlp['token'] not in IGNORED_SPECIAL_TOKENS
        ]

    # (0) Empty:
    if not token_logprobs:
        return 0

    if ignore_up_to_token:
        last_ignored = None
        for i, tlp in enumerate(token_logprobs):
            if tlp['token'].lower() == ignore_up_to_token:
                last_ignored = i
        if (last_ignored is not None) and len(token_logprobs) > i + 1:
            # There's at least one more token:
            token_logprobs = token_logprobs[i + 1:]

    removing_prefix = (ignore_prefix is not None)
    for tlp in token_logprobs:
        token = tlp['token'].lower()
        decoded += token
        decoded_lps.append(tlp['logprob'])
        top_tokens = tlp['top_tokens']
        top_logprobs = tlp['top_logprobs']
        if top_tokens is None:
            # If the model/service does not provide top tokens/log probabilities,
            # just the logprob for the singel top token (`token`), we simulate,
            # target probability for the computation to work. This is a little dirty,
            # (requires target to be single-token) but works just fine in most cases.
            assert top_logprobs is None
            if target == token: # after lowercasing,
                top_tokens = [token]
                top_logprobs = tlp['logprob']
            else:
                top_tokens = [token, target]
                token_lp = tlp['logprob']
                target_lp = np.log(1 - np.exp(token_lp))
                top_logprobs = [token_lp, target_lp]
        else:
            assert top_logprobs is not None

        top_tokens = [t.lower() for t in top_tokens]
        if removing_prefix:
            if ignore_prefix.startswith(decoded):
                continue
            if decoded.startswith(ignore_prefix):
                # Example:
                # ignore_prefix = 'english word: '
                # decoded = 'english word: straw'
                # token = ' straw'
                decoded         = decoded.removeprefix(ignore_prefix)  # 'straw'
                assert token.endswith(decoded)
                token           = decoded                               # 'straw'
                token_prefix    = token.removesuffix(decoded)           # ' '
                top_tokens      = [t.removeprefix(token_prefix) for t in top_tokens]
                decoded_lps     = decoded_lps[-1:]
            removing_prefix = False
        if not target.startswith(decoded):
            break
    else:
        if removing_prefix:
            # (1) Did not get past the ignored prefix:
            # Redo without trying to remove the ignored prefix:
            return _log_probs2prob(token_logprobs, target, temperature=temperature)

        # (2) `decoded` is a non-empty prefix of target
        assert decoded
        assert target.startswith(decoded), (target, decoded)

        # (2.1) We decoded the complete target: return its probability:
        valid_p = _log_prob2prob(np.sum(decoded_lps), temperature)
        if decoded == target:
            return valid_p

        # (2.2) We decoded incomplete prefix of the target. Should not happen.
        # Since we don't know, return 0.5 or `valid_p` if lower:
        return min(valid_p, 0.5)

    # (3) `decoded` deviates from target
    assert decoded.endswith(token)
    valid_prefix = decoded.removesuffix(token)
    if not target.startswith(valid_prefix):
        # (3.1) We must have started thinking that we're decoding the ignored prefix,
        # then it turned to deviate both from the ignored prefix and the target.
        # Redo without trying to remove the ignored prefix:
        assert ignore_prefix is not None
        assert ignore_prefix.startswith(valid_prefix)
        return _log_probs2prob(token_logprobs, target, temperature=temperature)

    valid_suffix = target.removeprefix(valid_prefix)
    valid_suffix_lps = [
        lp for t, lp in zip(top_tokens, top_logprobs) if valid_suffix.startswith(t)
        ]
    if not valid_suffix_lps:
        return 0
    valid_suffix_lp = (
        valid_suffix_lps[0] if (len(valid_suffix_lps) == 1) else
        np.log(np.sum(np.exp(valid_suffix_lps)))
        )
    return _log_prob2prob(
        np.sum([*decoded_lps[:-1], valid_suffix_lp]),
        temperature
        )


def prompting_logprobs2prob(
    logprobs: pd.Series, target: pd.Series | str,
    prompt: str,
    temperature: float | None = None
    ) -> pd.Series:

    ignore_prefix = None
    ignore_up_to_token = None

    if prompt in SOLVE_PROMPTS:
        ignore_prefix = _IGNORED_PREFIXES[prompt].lower()
    elif prompt == TRANSLATE_PROMPT:
        ignore_up_to_token = _TRANSLATE_SEP_TOKEN
    elif prompt not in ONE_ZERO_PROMPTS:
        raise Exception(f'Unknown prompt {prompt}')

    if isinstance(target, str):
        target = repeat(target)

    return pd.Series([
        _log_probs2prob(lp, t,
                        ignore_up_to_token=ignore_up_to_token,
                        ignore_prefix=ignore_prefix,
                        temperature=temperature
                        )
        for lp, t in zip(logprobs, target)
        ])


def prompts_and_models(prompts: list[str], models: list[str]) -> list[str]:
    return ['--'.join((p, m)) for p in prompts for m in models]


def _parse_rate_compare_output_row(output):
    parts = str(output if pd.notna(output) else '').split(',')
    parts = [part.strip() for part in parts]
    parts.extend([''] * (3 - len(parts)))
    vals = []
    for part in parts[:3]:
        try:
            val = int(part)
        except (TypeError, ValueError):
            val = 3
        vals.append(min(5, max(1, val)))
    return {'cn': vals[0], 'es': vals[1], 'de': vals[2]}


def _one_hot_rate_compare_probs(value):
    probs = np.zeros(5, dtype=float)
    probs[int(min(5, max(1, value))) - 1] = 1.0
    return probs


def _normalize_rate_compare_digit_lps(digit2lp):
    lps = np.array([digit2lp.get(str(d), -np.inf) for d in range(1, 6)])
    finite = np.isfinite(lps)
    if not finite.any():
        return _one_hot_rate_compare_probs(3)
    lps_finite = lps[finite]
    weights = np.exp(lps_finite - lps_finite.max())
    probs = np.zeros(5, dtype=float)
    probs[finite] = weights / weights.sum()
    return probs


def _rate_compare_row_soft_probs(logprobs, output):
    fallback = _parse_rate_compare_output_row(output)
    fallback_probs = {
        lang: _one_hot_rate_compare_probs(value)
        for lang, value in fallback.items()
        }
    if logprobs is None:
        return fallback_probs
    if isinstance(logprobs, str):
        if not logprobs.strip():
            return fallback_probs
        try:
            parsed = json.loads(logprobs)
        except (TypeError, ValueError, json.JSONDecodeError):
            return fallback_probs
    elif isinstance(logprobs, (list, tuple)):
        parsed = logprobs
    elif np.isscalar(logprobs):
        try:
            if pd.isna(logprobs):
                return fallback_probs
        except TypeError:
            pass
        return fallback_probs
    else:
        return fallback_probs

    digit_positions = []
    for token_lp in parsed:
        token = str(token_lp.get('token', '')).strip()
        if token not in {'1', '2', '3', '4', '5'}:
            continue
        digit2lp = {token: float(token_lp.get('logprob', -np.inf))}
        for top_token, top_lp in zip(
            token_lp.get('top_tokens', []),
            token_lp.get('top_logprobs', [])
            ):
            top_token = str(top_token).strip()
            if top_token in {'1', '2', '3', '4', '5'}:
                digit2lp[top_token] = float(top_lp)
        digit_positions.append(_normalize_rate_compare_digit_lps(digit2lp))
        if len(digit_positions) == 3:
            break

    if len(digit_positions) != 3:
        return fallback_probs

    return {
        'cn': digit_positions[0],
        'es': digit_positions[1],
        'de': digit_positions[2]
        }


def split_rate_compare_soft_probs(logprobs, outputs):
    if isinstance(logprobs, pd.DataFrame):
        logprobs = logprobs.iloc[:, 0]
    if isinstance(outputs, pd.DataFrame):
        outputs = outputs.iloc[:, 0]
    lang2probs = {lang: [] for lang in ('cn', 'es', 'de')}
    for lp, out in zip(logprobs, outputs):
        row_probs = _rate_compare_row_soft_probs(lp, out)
        for lang in ('cn', 'es', 'de'):
            lang2probs[lang].append(row_probs[lang])
    return {
        lang: np.vstack(probs)
        for lang, probs in lang2probs.items()
        }


def concat_feature_columns(df, new_cols):
    if isinstance(new_cols, dict):
        if not new_cols:
            return df
        new_cols = pd.DataFrame(new_cols, index=df.index)
    return pd.concat((df, new_cols), axis=1)
