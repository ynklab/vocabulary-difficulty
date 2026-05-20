import sys
import json
import re
import shlex
from itertools import repeat, chain
from time import sleep
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from collections.abc import Iterable
from kvl import (
    read_subset, read_data_cv, SUBSETS, L1_CODES, LANG2NAME, ID_COL, spaced_clue
    )
from prompt_util import SOLVE_PROMPTS
from tubelex_util import tubelex_lr_error

from tqdm import tqdm
from openai import OpenAI
from openai._exceptions import (
    RateLimitError,
    APIStatusError,
    APITimeoutError
    )
import config

import nltk
from nltk.corpus import cmudict


DEFAULT_SPLITS_PATH = 'data/cv-split-ids-5.json'

DIFFICULTY_PROMPTS2SHOT_LABELS = {
    'zs-difficulty': [],
    '3s-difficulty': [1, 3, 5],
    '5s-difficulty': [1, 2, 3, 4, 5]
    }
N_LABELS = 5
EXAMPLE_HARDNESS2WEIGHTS = {
    'any': (0,),
    'easy': (1,),
    'hard': (-1,),
    'both': (1, -1,)
    }

CMU = None


def cmu_phonemes(word: str, remove_stress: bool = True):
    global CMU
    if CMU is None:
        try:
            CMU = cmudict.dict()
        except LookupError:
            nltk.download('cmudict')
            CMU = cmudict.dict()

    prons = CMU.get(word.lower())
    if not prons:
        return ""

    if remove_stress:
        return " ".join([[re.sub(r"\d", "", ph) for ph in pron] for pron in prons][0])
    return " ".join(prons[0])


def scores2continuous_labels_scale_factor(scores, reverse: bool, min=None, max=None):
    '''
    The labels are centers of N_LABELS bins.

    With N_LABELS=5, we convert like this:

    scores = pd.Series([5, 10, 20, 30, 40, 50, 55, 18.5])

    scores2labels(scores, reverse=False): [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5, 1.85], 0.1
    scores2labels(scores, reverse=True): [5.5, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 4.15], -0.1

    We also return a signed scale factor.
    '''

    s_min = min if (min is not None) else scores.min()
    s_max = max if (max is not None) else scores.max()
    if reverse:
        (s_min, s_max) = (s_max, s_min)

    a = s_min + (s_max - s_min) / (N_LABELS * 2)
    d = (s_max - s_min) * (N_LABELS - 1) / N_LABELS

    scale_factor = (N_LABELS - 1) / d
    return (((scores - a) * scale_factor) + 1, scale_factor)


SPELLING_DIFF_PROMPTS = [
    'spelling_diff_without_l1_word',
    'spelling_diff_with_l1_word'
    ]

PROMPTS = [
    *SOLVE_PROMPTS, *DIFFICULTY_PROMPTS2SHOT_LABELS,
    'calque', 'transliteration',
    'lexical_ambiguity',
    *SPELLING_DIFF_PROMPTS,
    'morphtran'
    ]  # Add more as needed

SHORT_OUTPUT_PROMPTS = {
    *SOLVE_PROMPTS, *DIFFICULTY_PROMPTS2SHOT_LABELS,
    *SPELLING_DIFF_PROMPTS,
    'morphtran'
    }  # Add more as needed


def get_max_tokens(prompt: str) -> int:
    return 16 if (prompt in SHORT_OUTPUT_PROMPTS) else 64


def format_examples(
    *,
    l1_label: str,
    context_label: str,
    english_label: str,
    l1_words,
    l1_contexts,
    en_words,
    difficulties: Iterable[int] | None = None
    ) -> str:
    iter_opt_difficulties = (
        repeat('', len(l1_words)) if (difficulties is None) else
        (f'Difficulty: {d}\n' for d in difficulties)
        )
    examples = ''.join(
        f'''\
{l1_label}: {l1_word}
{context_label}: {l1_context}
Clue: {spaced_clue(en_word)}
{english_label}: {en_word}
{opt_difficulty}
'''
        for l1_word, l1_context, en_word, opt_difficulty in zip(
            l1_words, l1_contexts, en_words, iter_opt_difficulties
            )
        )
    if not examples:
        raise ValueError('No few-shot examples available for solve prompts.')
    return examples


def get_prompt(
    prompt: str,
    l1: str,
    ex_l1_words: Iterable[str],
    ex_l1_contexts: Iterable[str],
    ex_en_words: Iterable[str],
    ex_difficulties: Iterable[int] | None,
    l1_word: str, l1_context: str, en_word: str,
    en_pos: str,
    all_l1_words: dict[str, str] | None = None
    ) -> str:

    l1_name = LANG2NAME[l1]

    def first_example(values):
        # Used for translate, we ignore the other values
        return values.iloc[0] if hasattr(values, 'iloc') else next(iter(values))

    def solve_examples():
        # Used for solve and long-solve
        return format_examples(
            l1_label=f'{l1_name} word',
            context_label=f'{l1_name} context',
            english_label='English word',
            l1_words=ex_l1_words,
            l1_contexts=ex_l1_contexts,
            en_words=ex_en_words
            )

    if prompt == 'solve':
        clue = spaced_clue(en_word)
        return f'''\
You are bilingual in {l1_name} and English and your task is to find the best English \
translation for a {l1_name} word given a context and constraints. The constraints are \
given in the form of a clue, e.g., "b _ _ _", meaning that the word starts with \
the (upper or lower case) letter B and has 4 letters. You must give a single \
English word in dictionary form (lemma) as a response.

{solve_examples()}\
{l1_name} word: {l1_word}
{l1_name} context: {l1_context}
Clue: {clue}
English word:'''

    if prompt == 'short-solve':
        # Suggested by ChatGPT-5.2 instructed to "save tokens"
        clue = spaced_clue(en_word)
        return f'''\
Find the best English lemma translating the {l1_name} word in context.
The word must match the clue pattern (letters + "_").
Output ONE word only.

{l1_name}: {l1_word}
Context: {l1_context}
Clue: {clue}
English:'''

    if prompt == 'terse-solve':
        # Also suggested by ChatGPT-5.2 instructed to "save tokens"
        clue = spaced_clue(en_word)
        return f'''\
Translate to an English word fitting the clue.

{l1_name}: {l1_word}
Context: {l1_context}
Clue: {clue}
English:'''

    if prompt == 'long-solve':
        # The above prompt rewritten by ChatGPT-5.2 instructed to
        # "improve it/make it clearer".
        clue = spaced_clue(en_word)
        return f'''\
You are bilingual in {l1_name} and English.

TASK
Given a word in {l1_name}, its usage context, and a spelling clue, find the single \
best English translation that fits BOTH the meaning and the spelling constraint.

INPUTS
- {l1_name} word: a single word to translate
- {l1_name} context: a sentence showing how the word is used
- Clue: a pattern such as "b _ _ _", where:
  * the first letter is indicated (case-insensitive)
  * "_" indicates subsequent unknown letter
  * the total number of letters must match exactly

OUTPUT REQUIREMENTS
- Output EXACTLY ONE English word
- The word must be:
  * a dictionary form (lemma)
  * a single token (no spaces, hyphens, or punctuation)
  * consistent with the context
  * consistent with the clue
- Do NOT include explanations, alternatives, quotes, or extra text.

EXAMPLES
{solve_examples()}\
NOW SOLVE
{l1_name} word: {l1_word}
{l1_name} context: {l1_context}
Clue: {clue}
English word:'''

    elif prompt in DIFFICULTY_PROMPTS2SHOT_LABELS:
        examples = format_examples(
            l1_label=f'{l1_name} word',
            context_label=f'{l1_name} context',
            english_label='English word',
            l1_words=ex_l1_words,
            l1_contexts=ex_l1_contexts,
            en_words=ex_en_words,
            difficulties=ex_difficulties,
            )
        clue = spaced_clue(en_word)
        return f'''\
You are an English language teacher teaching learners whose native language is \
{l1_name}. Your task is to rate the difficulty of a vocabulary test item \
for native {l1_name} speakers learning English.

The test item consists of:
- a {l1_name} word,
- a {l1_name} context,
- a clue indicating the first letter and word length of the English word,
- the target English word, which is the only correct answer.

Letter case does not matter. The learners are likely to respond with synonyms \
or misspellings to some items, but such responses are considered incorrect. Treat this \
as increasing the difficulty.

Consider learners from beginner to advanced levels, weighting the intermediate learner \
most heavily. Rate how difficult the item is on a scale from 1 to 5:
1 = very easy (almost everybody answers correctly)
5 = very difficult (almost nobody answers correctly)

Output exactly one digit (1, 2, 3, 4, or 5). Do not include any other text.

{examples}\
{l1_name} word: {l1_word}
{l1_name} context: {l1_context}
Clue: {clue}
English word: {en_word}
Difficulty:'''

    elif prompt == 'translate':
        ex_l1_word = first_example(ex_l1_words)
        ex_l1_context = first_example(ex_l1_contexts)
        ex_en_word = first_example(ex_en_words)
        ex_en_context = EX_L1_CONTEXT_TRANSLATIONS[(l1, ex_l1_context)]
        ex_letter = ex_en_word[0].upper()
        letter = en_word[0].upper()
        return f'''\
Your are a professional translator and your task is to translate a sentence given a \
hint for a keyword. Do not focus only on the hint. The keyword must fit the context \
perfectly. After translating the sentence you will also give the translation of the \
keyword in dictionary form (lemma) separated by a slash (/).

{l1_name} sentence: {ex_l1_context}
Hint: The translation of the keyword "{ex_l1_word}" starts with the letter {ex_letter}.
Translated sentence / keyword: {ex_en_context} / {ex_en_word}

{l1_name} sentence: {l1_context}
Hint: The translation of the keyword "{l1_word}" starts with the letter {letter}.
Translated sentence / keyword:'''

    elif prompt == 'calque':
        ex_calque_l1, ex_calque_en = EX_L1_CALQUE[l1]
        return f'''\
You are a bilinguistics expert. \

TASK
Given a {l1_name} item and an English item, decide whether \
there exists a best-matching candidate in the {l1_name} item that is a component-by-component (morpheme-level) translation of the English item.

A component-by-component mapping means that the meaningful parts \
(words, roots, prefixes, or suffixes) of the English item are directly translated \
into corresponding meaningful parts in the {l1_name} item.

Procedure (internal; do NOT output these steps):
1) If the {l1_name} item contains multiple candidates, select exactly ONE candidate: the one that aligns best component-wise with the English form.
2) Judge ONLY that selected candidate for component-by-component mapping.

OUTPUT REQUIREMENTS
- Output "1" if the selected best candidate is a component-by-component mapping; otherwise output "0".
- Output MUST be exactly one character: 1 or 0.
- Do NOT include explanations, alternatives, quotes, or extra text.

EXAMPLE
{l1_name} item: {ex_calque_l1}
English item: {ex_calque_en}
Is word-for-word mapping: 1

NOW DECIDE
{l1_name} item: {l1_word}
English item: {en_word}
Is word-for-word mapping:'''

    elif prompt == 'morphtran':
        return f'''\
You are a linguist and your task is to decide whether an English word is a \
morpheme-for-morpheme translation of any of the given {l1_name} equivalents. \
The morpheme-for-morpheme mapping must be 1:1. 1:N or other mappings do not count. \
Single morpheme translations or simple borrowings/cognates do not count either. \
Respond only with YES or NO.

wave/ola: NO (reason: single morpheme)
ecosystem/ecosistema: NO (reason: simple cognate)
hotdog/perro caliente: YES (reason: hot=caliente, dog=perro)
stare/mirar fijamente: NO (reason: not a 1:1 mapping)
{en_word}/{l1_word}:'''

    elif prompt == 'transliteration':
        ex_trans_l1, ex_trans_en = EX_L1_TRANSLITERATION[l1]
        return f'''\
You are a linguistics expert specializing in transliterations (phonetic renderings of \
words from one language into another) \

TASK
Given a {l1_name} item and an English item, decide whether \
the {l1_name} item is plausibly a transliteration of the English source (i.e., primarily based on pronunciation rather than meaning).

Procedure (internal; do NOT output these steps):
1) If the {l1_name} item contains multiple candidates, select exactly ONE candidate: \
   the one that best approximates the pronunciation of the English item.
2) Analyze the phonetic structure of both the {l1_name} item and the English item.
3) Determine if the {l1_name} item closely approximates the pronunciation of the English
    item, considering typical phonetic adaptations between the two languages.

OUTPUT REQUIREMENTS
- Output "1" if the target is plausibly a transliteration from the given English source. "0" otherwise.
- Output MUST be exactly one character: 1 or 0.
- Do NOT include explanations, alternatives, quotes, or extra text.

EXAMPLE
{l1_name} item: {ex_trans_l1}
English source: {ex_trans_en}
Is transliteration: 1

NOW DECIDE
{l1_name} item: {l1_word}
English source: {en_word}
Is transliteration:'''

    elif prompt == 'lexical_ambiguity':
        ex_en_word = EX_AMBIGUOUS_WORD
        ex_easy_word_l1, ex_easy_context_l1 = EX_L1_AMBIGUOUS[l1][0]
        ex_hard_word_l1, ex_hard_context_l1 = EX_L1_AMBIGUOUS[l1][1]
        return f'''\
You are a language education expert.

TASK
Given:
- an English word form (the "English word"),
- an L1 gloss/translation (the "{l1_name} item"),
- and the L1 usage context sentence (the "{l1_name} context"),
decide whether the English word, when used to express the meaning suggested by the L1 item + context,
meets BOTH conditions:

A) Lexical ambiguity: the English word has multiple established senses that share the same form
   (polysemy or homonymy), such that another common sense could plausibly be activated/confused.

B) Unfamiliarity for L2 learners: in this meaning/usage, the English word is likely to be unfamiliar
   or challenging for typical second-language learners (e.g., less frequent sense, idiomatic/figurative,
   domain-specific usage, nonliteral extension).

OUTPUT REQUIREMENTS
- Output "1" if BOTH conditions (A and B) are met; otherwise output "0".
- Output MUST be exactly one character: 1 or 0.
- Do NOT include explanations, alternatives, quotes, or extra text.

EXAMPLE 1
English word: {ex_en_word}
{l1_name} item: {ex_easy_word_l1}
{l1_name} context: {ex_easy_context_l1}
Is the English word ambiguous and unfamiliar: 0

EXAMPLE 2
English word: {ex_en_word}
{l1_name} item: {ex_hard_word_l1}
{l1_name} context: {ex_hard_context_l1}
Is the English word ambiguous and unfamiliar: 1

NOW DECIDE
English word: {en_word}
{l1_name} item: {l1_word}
{l1_name} context: {l1_context}
Is the English word ambiguous and unfamiliar:'''

    elif prompt == 'spelling_diff_without_l1_word':
        # assert en_pron, 'en_pron is required for pronounce_difficulty prompt'
        en_pron = cmu_phonemes(en_word)
        return f'''\
TASK
You are required to rate English spelling difficulty on a 1–5 scale, where 1 = very easy and 5 = very difficult.
You will be given English pronunciation.
Evaluate how difficult it would be for learners with Chinese, Spanish and German L1 backgrounds to spell an English word with that pronunciation correctly when they know the translation in their native language.

OUTPUT REQUIREMENTS
- Output exactly one digit (1, 2, 3, 4, or 5) for each L1, separated by commas, in the order of Chinese, Spanish, German. Do not include any other text.

EXAMPLE 1
English pronunciation: 'K Y UW'
Result:
Chinese: 5, Spanish: 4, German: 4

EXAMPLE 2
English pronunciation: 'SH UW'
Result:
Chinese: 1, Spanish: 2, German: 1


NOW DECIDE
English pronunciation: {en_pron}
Result:'''

    elif prompt == 'spelling_diff_with_l1_word':
        en_pron = cmu_phonemes(en_word)
        if all_l1_words is None:
            raise ValueError(
                'all_l1_words is required for spelling_diff_with_l1_word'
                )
        easy_cn, _, easy_pron, easy_cn_score = EX_L1_SPELL_EASY['cn']
        easy_es, _, _, easy_es_score = EX_L1_SPELL_EASY['es']
        easy_de, _, _, easy_de_score = EX_L1_SPELL_EASY['de']
        hard_cn, _, hard_pron, hard_cn_score = EX_L1_SPELL_DIFFICULT['cn']
        hard_es, _, _, hard_es_score = EX_L1_SPELL_DIFFICULT['es']
        hard_de, _, _, hard_de_score = EX_L1_SPELL_DIFFICULT['de']
        return f'''\
TASK
You are required to rate English spelling difficulty on a 1–5 scale, where 1 = very easy and 5 = very difficult.
You will be given English pronunciation and the target word's translation in Chinese, Spanish, and German.
Evaluate how difficult it would be for learners with Chinese, Spanish, and German L1 backgrounds to spell the English word with that pronunciation correctly when they know the translation in their native language.

OUTPUT REQUIREMENTS
- Output exactly one digit (1, 2, 3, 4, or 5) for each L1, separated by commas, in the order of Chinese, Spanish, German.
- Do not include any other text.

EXAMPLE 1
English pronunciation: '{hard_pron}'
Chinese: {hard_cn}
Spanish: {hard_es}
German: {hard_de}
Result: {hard_cn_score},{hard_es_score},{hard_de_score}

EXAMPLE 2
English pronunciation: '{easy_pron}'
Chinese: {easy_cn}
Spanish: {easy_es}
German: {easy_de}
Result: {easy_cn_score},{easy_es_score},{easy_de_score}

NOW DECIDE
English pronunciation: {en_pron}
Chinese: {all_l1_words['cn']}
Spanish: {all_l1_words['es']}
German: {all_l1_words['de']}
Result:'''

    raise Exception('Unimplemented prompt "{prompt}"')


def model_short_name(m: str) -> str:
    # Leave out vendor for DeepInfra models:
    m = re.sub(r'^.*/', '', m)
    # Leave out what looks like a date at the end & lowercase:
    return re.sub(r'-[0-9]{4}-[0-9]{2}-[0-9]{2}$', '', m).lower()


def logprobs2json(logprobs):
    return json.dumps([{
        'token': lp.token,
        'logprob': lp.logprob,
        # top_logprobs is None for DeepInfra models:
        'top_tokens': (
            None if (lp.top_logprobs is None) else
            [tlp.token for tlp in lp.top_logprobs]
            ),
        'top_logprobs': (
            None if (lp.top_logprobs is None) else
            [tlp.logprob for tlp in lp.top_logprobs]
            )
        } for lp in logprobs.content])


def response_logprobs2json(logprobs):
    return json.dumps([{
        'token': lp['token'],
        'logprob': lp['logprob'],
        'top_tokens': [tlp['token'] for tlp in lp['top_logprobs']],
        'top_logprobs': [tlp['logprob'] for tlp in lp['top_logprobs']],
        } for lp in logprobs])


EX_WORD = 'strawberry'
EX_L1_CONTEXT_TRANSLATIONS = {
    ('cn', '她做了草莓蛋糕。'): 'She made a strawberry cake.',
    ('de', 'Ich mag keine Erdbeeren.'): 'I don\'t like strawberries.',
    ('es', 'Voy a tomar un trozo de tarta de fresa.'):
        'I\'m going to have a piece of strawberry cake.'
    }
EX_L1_CALQUE = {
    'cn': ('热狗', 'hot dog'),
    'de': ('Wolkenkratzer', 'skyscraper'),
    'es': ('Luna de miel', 'honeymoon')
    }
EX_L1_TRANSLITERATION = {
    'cn': ('咖啡', 'coffee'),
    'de': ('Computer', 'Computer'),
    'es': ('Fútbol', 'Football')
    }
EX_AMBIGUOUS_WORD = 'bank'
EX_L1_AMBIGUOUS = {
    'cn': [
        ('银行', '我把钱存在银行里。'),
        ('岸', '我们坐在河岸上。')
        ],
    'de': [
        ('Bank', 'Ich habe das Geld auf die Bank eingezahlt.'),
        ('Ufer', 'Wir saßen am Flussufer.')
        ],
    'es': [
        ('banco', 'Deposité el dinero en el banco.'),
        ('orilla', 'Nos sentamos en la orilla del río.')
        ]
    }

EX_L1_SPELL_EASY = {
    'cn': ['袋鼠', 'kangaroo', 'K AE NG G ER UW', 1],
    'es': ['chocolate', 'chocolate', 'CH AO K L AH T', 1],
    'de': ['Haus', 'house', 'HH AW S', 1]
    }

EX_L1_SPELL_DIFFICULT = {
    'cn': ['队列', 'queue', 'K Y UW', 5],
    'es': ('cuchillo', 'knife', 'N AY F', 4),
    'de': ('Insel', 'island', 'AY L AH N D', 4),
    }


def prompting_filename(args: argparse.Namespace) -> str:
    return f'{args.prompt}--{model_short_name(args.model)}{args.suffix}.csv'


def prompting_output_path(
    outdir: Path, output_scope: str, args: argparse.Namespace
    ) -> Path:
    return outdir / output_scope / prompting_filename(args)


def existing_prompting_output_path(path: Path):
    if path.exists():
        return path
    compressed = Path(f'{path}.xz')
    return compressed if compressed.exists() else None


def mode_scope_tag(args: argparse.Namespace, legacy_subsets_mode: bool) -> str:
    if args.final_data:
        raw = 'final-data-test'
    elif legacy_subsets_mode:
        raw = f'subsets-{"-".join(args.subsets)}'
    else:
        raw = f'cv-{args.cv_mode}-{Path(args.splits).stem}'
    return re.sub(r'[^A-Za-z0-9._-]+', '-', raw)


def checkpoint_results_path(scope_tag: str, args: argparse.Namespace) -> Path:
    return (
        Path('batch') /
        f'results-{scope_tag}-{args.prompt}--'
        f'{model_short_name(args.model)}{args.suffix}.json'
        )


def save_checkpoint_results(path: Path, results_by_scope: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'results_by_scope': {}}
    for scope, scope_results in results_by_scope.items():
        payload['results_by_scope'][scope] = {
            str(item_id): dict(item_results)
            for item_id, item_results in scope_results.items()
            }
    with open(path, 'w') as f:
        json.dump(payload, f, ensure_ascii=False)


def load_checkpoint_results(path: Path) -> dict[str, dict[int, dict]]:
    if not path.exists():
        raise ValueError(
            f'--continue requires existing results at {path}, but none found.'
            )
    with open(path, 'r') as f:
        payload = json.load(f)
    raw_results = payload.get('results_by_scope')
    if not isinstance(raw_results, dict) or not raw_results:
        raise ValueError(
            f'--continue found no usable results in {path}.'
            )
    parsed = {}
    for scope, scope_results in raw_results.items():
        if not isinstance(scope_results, dict):
            continue
        parsed_scope = {}
        for raw_item_id, item_results in scope_results.items():
            if not isinstance(item_results, dict):
                continue
            try:
                item_id = int(raw_item_id)
            except (TypeError, ValueError):
                continue
            parsed_scope[item_id] = dict(item_results)
        if parsed_scope:
            parsed[scope] = parsed_scope
    if not parsed:
        raise ValueError(
            f'--continue found no usable item-level results in {path}.'
            )
    return parsed


def continue_command() -> str:
    argv = list(getattr(sys, 'orig_argv', []) or [sys.executable, *sys.argv])
    if '--continue' not in argv:
        argv.append('--continue')
    return shlex.join(argv)


def maybe_write_whole_from_complement(
    *,
    args: argparse.Namespace,
    outdir: Path,
    current_scope: str,
    current_df: pd.DataFrame,
    all_item_ids: set[int]
    ):
    complement_scope = {
        'first': 'remaining',
        'remaining': 'first'
        }.get(current_scope)
    if complement_scope is None:
        return

    complement_path = existing_prompting_output_path(
        prompting_output_path(outdir, complement_scope, args)
        )
    if complement_path is None:
        print(
            f'Complementary predictions not found for {complement_scope}; '
            f'skipping whole output.',
            file=sys.stderr
            )
        return

    complement_df = pd.read_csv(complement_path, index_col=ID_COL)
    if list(current_df.columns) != list(complement_df.columns):
        print(
            'Complementary predictions have different columns; '
            'skipping whole output.',
            file=sys.stderr
            )
        return

    merged = pd.concat((current_df, complement_df))
    if merged.index.duplicated().any():
        dup_ids = merged.index[merged.index.duplicated()].tolist()[:10]
        print(
            'Complementary predictions contain duplicate item_id values; '
            f'skipping whole output. Examples: {dup_ids}',
            file=sys.stderr
            )
        return

    merged_ids = set(merged.index.tolist())
    missing = sorted(all_item_ids - merged_ids)
    extra = sorted(merged_ids - all_item_ids)
    if missing or extra:
        print(
            'Complementary predictions do not cover full data; '
            f'skipping whole output. missing={missing[:10]} extra={extra[:10]}',
            file=sys.stderr
            )
        return

    whole_path = prompting_output_path(outdir, 'whole', args)
    whole_path.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_index().to_csv(whole_path)
    print(f'Writing whole results to {whole_path}.', file=sys.stderr)


def main(args: argparse.Namespace):
    np.random.seed(args.seed)

    legacy_subsets_mode = args.subsets is not None
    if args.final_data and legacy_subsets_mode:
        raise ValueError('--final-data is not supported with --subsets.')
    if args.final_data and args.cv_mode != 'whole':
        raise ValueError(
            '--final-data does not support --cv-mode first/remaining.'
            )
    if args.final_data and args.splits != DEFAULT_SPLITS_PATH:
        raise ValueError(
            '--final-data does not use custom --splits. '
            f'Use the default ({DEFAULT_SPLITS_PATH}) or omit --splits.'
            )
    output_scope2splits = {}
    all_item_ids = None
    lang2score_min = {}
    lang2score_max = {}
    is_gpt52 = 'gpt-5.2' in args.model

    if is_gpt52 and not (args.batch or (args.from_batches is not None)):
        print(
            'Warning: Using GPT-5.2 with real time API (not batch API) results in '
            'only 5 logprobs instead of 20. Will start anyway in 5 seconds.',
            file=sys.stderr
            )
        for i in range(5):
            print('.', file=sys.stderr)
            sleep(1)
        print('.', file=sys.stderr)

    example_pool = None

    if legacy_subsets_mode:
        subset2data = {subset: read_subset(subset) for subset in SUBSETS}
        example_pool = pd.concat(
            (subset2data['train'], subset2data['dev']),
            ignore_index=True
            )
        for lang in args.l1s:
            # We use global min/max for the scale for consistency
            lang2score_min[lang] = example_pool[f'{lang}_GLMM_score'].min()
            lang2score_max[lang] = example_pool[f'{lang}_GLMM_score'].max()
        print(
            'Warning: --subsets uses legacy train/dev mode. '
            'CV mode (--cv-mode/--splits) is ignored.',
            file=sys.stderr
            )
        for subset in args.subsets:
            complement = 'dev' if subset == 'train' else 'train'
            data_train = subset2data[complement]
            data_dev = subset2data[subset]
            output_scope2splits[subset] = [{
                'label': subset,
                'request_tag': subset,
                'train': data_train,
                'dev': data_dev
                }]
    else:
        data_cv = read_data_cv(
            (None if args.final_data else args.splits),
            train=('full' if args.final_data else None),
            eval=('test' if args.final_data else None)
            )
        data = data_cv.data
        if args.final_data:
            if len(data_cv.cv) != 1:
                raise ValueError(
                    '--final-data expected exactly one train/eval split.'
                    )
            example_pool = data.iloc[data_cv.cv[0].train]
        else:
            example_pool = data
        for lang in args.l1s:
            # We use global min/max for the scale for consistency
            lang2score_min[lang] = example_pool[f'{lang}_GLMM_score'].min()
            lang2score_max[lang] = example_pool[f'{lang}_GLMM_score'].max()
        selected_splits = list(enumerate(data_cv.cv, 1))
        output_scope = 'test' if args.final_data else args.cv_mode
        if args.final_data:
            # Keep the single full->test split as-is.
            pass
        elif args.cv_mode == 'first':
            selected_splits = selected_splits[:1]
        elif args.cv_mode == 'remaining':
            selected_splits = selected_splits[1:]
        else:
            assert args.cv_mode == 'whole'
        if not selected_splits:
            raise ValueError(
                f'No splits selected for cv-mode={args.cv_mode}. '
                'Need at least 2 folds for remaining mode.'
                )
        output_scope2splits[output_scope] = []
        for fold_i, split in selected_splits:
            data_train = data.iloc[split.train]
            data_dev = data.iloc[split.dev]
            output_scope2splits[output_scope].append({
                'label': (
                    f'fold {fold_i}'
                    if not args.final_data else
                    'test'
                    ),
                'request_tag': (
                    f'{args.cv_mode}-fold{fold_i}'
                    if not args.final_data else
                    'final-data'
                    ),
                'train': data_train,
                'dev': data_dev
                })
        all_item_ids = set(data[ID_COL].tolist())
    if example_pool is None:
        raise ValueError('Unable to initialize few-shot example pool.')
    examples = example_pool[example_pool['en_target_word'] == EX_WORD]
    if examples.empty:
        raise ValueError(
            f'No few-shot examples found for en_target_word={EX_WORD}.'
            )

    if args.dry_run_hardness_hist:
        for lang in args.l1s:
            _, scale_factor = scores2continuous_labels_scale_factor(
                example_pool[f'{lang}_GLMM_score'],
                min=lang2score_min.get(lang),
                max=lang2score_max.get(lang),
                reverse=True
                )
            scaled_lr_err = tubelex_lr_error(example_pool, lang) * scale_factor
            hardness = np.abs(scaled_lr_err)
            counts, bin_edges = np.histogram(hardness, bins=10)
            print(f'{lang} hardness histogram (n={len(hardness)}):')
            for i, (left, right, count) in enumerate(
                zip(bin_edges[:-1], bin_edges[1:], counts),
                start=1
                ):
                print(
                    f'  bin {i:02d}: [{left:.6f}, {right:.6f}) -> {int(count)}'
                    )
            print(f'  total: {int(counts.sum())}')
            print()
        return

    max_tokens = args.max_tokens or get_max_tokens(args.prompt)
    assert max_tokens >= 16  # API requirement

    api_model = args.model
    if args.deepinfra:
        client = OpenAI(
            base_url='https://api.deepinfra.com/v1/openai',
            api_key=config.DEEPINFRA_API_KEY
            )
    elif args.deepseek:
        client = OpenAI(
            base_url='https://api.deepseek.com',
            api_key=config.DEEPSEEK_API_KEY
            )
        assert model_short_name(args.model) == 'deepseek-v3.2'
        api_model = 'deepseek-chat'
    else:
        client = OpenAI(api_key=config.API_KEY)

    outdir = Path(args.output_directory) / 'prompting'
    logdir = Path(args.log_directory)
    logdir.mkdir(parents=True, exist_ok=True)

    usage_by_lang = defaultdict(Counter)
    diagnostics = {
        'prompt': args.prompt,
        'legacy_subsets_mode': legacy_subsets_mode,
        'subsets': args.subsets,
        'final_data': args.final_data,
        'cv_mode': args.cv_mode,
        'splits': args.splits,
        'l1s': args.l1s,
        'model': api_model,
        'system_fingerprint': None,
        'usage': usage_by_lang,
        }

    if args.rebatch_missing and (args.from_batches is None):
        raise Exception('--rebatch-missing requires --from-batches.')
    if args.continue_mode and args.batch:
        raise ValueError('--continue cannot be used with --batch.')
    if args.continue_mode and (args.from_batches is not None):
        raise ValueError('--continue cannot be used with --from-batches.')
    if args.continue_mode and args.rebatch_missing:
        raise ValueError('--continue cannot be used with --rebatch-missing.')

    scope_tag = mode_scope_tag(args, legacy_subsets_mode=legacy_subsets_mode)
    checkpoint_path = checkpoint_results_path(scope_tag, args)
    checkpoint_results_by_scope = (
        load_checkpoint_results(checkpoint_path)
        if args.continue_mode else
        {}
        )
    responses = {}
    requests = []
    if args.from_batches is not None:
        bpath = Path('batch')
        bpath.mkdir(parents=True, exist_ok=True)

        batch_ids: list[str] = args.from_batches
        if not batch_ids:
            path = (
                bpath /
                f'batch_id-{scope_tag}-{args.prompt}--'
                f'{model_short_name(args.model)}{args.suffix}.txt'
                )
            with open(path, 'r') as f:
                batch_id = f.read()
                assert batch_id
                batch_ids = [batch_id]
        assert isinstance(batch_ids, list)
        print(f'Batch IDs: {batch_ids}')

        diagnostics['batch_ids'] = batch_ids

        for batch_id in batch_ids:
            batch = client.batches.retrieve(batch_id)
            if batch.status != 'completed':
                print(batch)
                raise Exception(f'Batch {batch_id} not completed yet: {batch.status}.')

            if batch.output_file_id:
                content = client.files.content(batch.output_file_id)
                it_outputs = (json.loads(line) for line in content.text.splitlines())
                for o in it_outputs:
                    cid = o['custom_id']
                    if cid not in responses:
                        responses[cid] = o['response']  # Earlier batch wins
            elif batch.error_file_id:
                errors = client.files.content(batch.error_file_id)
                print(errors.text)
                sys.exit(1)
            else:
                raise RuntimeError(
                    'Batch completed but no output or error file found.'
                    )

    batch_missing = []

    def requests_append(request_id, prompt):
        requests.append({
            'custom_id': request_id,
            'method': 'POST',
            'url': '/v1/responses',
            'body': {
                'model': api_model,
                'input': prompt,
                'temperature': 0,
                'max_output_tokens': max_tokens,
                'include': ['message.output_text.logprobs'],
                'top_logprobs': 20
                }
            })

    example_labels  = DIFFICULTY_PROMPTS2SHOT_LABELS.get(args.prompt)
    for output_scope, splits in output_scope2splits.items():
        results = defaultdict(dict)
        loaded_scope_results = checkpoint_results_by_scope.get(output_scope, {})
        for item_id, item_results in loaded_scope_results.items():
            results[item_id] = dict(item_results)
        for split in splits:
            data_train  = split['train']    # few-shot candidate pool (stub for now)
            data_dev    = split['dev']
            if data_train.empty:
                raise ValueError(f'No train data for {output_scope}/{split["label"]}')
            for lang in args.l1s:
                if example_labels is not None:
                    # replace `examples` with `n_examples` from labeled full pool
                    dev_labels, scale_factor = scores2continuous_labels_scale_factor(
                        example_pool[f'{lang}_GLMM_score'],
                        min=lang2score_min.get(lang),
                        max=lang2score_max.get(lang),
                        reverse=True    # scores are facility, not difficulty
                        )
                    # positive scaled_lr_err = higher difficulty is predicted
                    scaled_lr_err = (
                        tubelex_lr_error(example_pool, lang) * scale_factor
                        )

                    hardness_weigths = EXAMPLE_HARDNESS2WEIGHTS[args.example_hardness]
                    ex_ilocs = list(chain.from_iterable(
                        (
                            (
                                # Close to the integer label & easy to predict:
                                2 * (dev_labels - ex_label).abs() +
                                hw * np.abs(scaled_lr_err)      # scaled to same units
                                ).argmin()
                            for ex_label in example_labels
                            )
                        for hw in hardness_weigths
                        ))
                    examples = example_pool.iloc[ex_ilocs]
                    ex_difficulties = example_labels * len(hardness_weigths)
                    if args.log_examples:
                        print(f'{lang} EXAMPLES:  {examples}', file=sys.stderr)
                        print(f'{lang} LR ERRORS: {scaled_lr_err[ex_ilocs]} '
                              f'MAE={np.abs(scaled_lr_err[ex_ilocs]).mean():.2f}',
                              file=sys.stderr)
                        print(f'{lang} LABELS:    {list(dev_labels.iloc[ex_ilocs])}',
                              file=sys.stderr)
                else:
                    ex_difficulties = None
                ex_l1_words = examples[f'{lang}_L1_source_word']
                ex_l1_contexts = examples[f'{lang}_L1_context']
                ex_en_words = examples['en_target_word']
                inputs = list(data_dev[[
                    ID_COL, 'en_target_word', 'en_target_pos',
                    f'{lang}_L1_source_word', f'{lang}_L1_context',
                    'cn_L1_source_word', 'es_L1_source_word', 'de_L1_source_word'
                    ]].itertuples(index=False))
                if args.limit is not None:
                    inputs = inputs[:args.limit]
                if args.item_ids:
                    wanted_item_ids = set(args.item_ids)
                    inputs = [
                        (id_value, *fields)
                        for id_value, *fields in inputs
                        if id_value in wanted_item_ids
                        ]
                action = 'Creating requests' if args.batch else 'Prompting'
                for (item_id, en_word, en_pos, l1_word, l1_context,
                     cn_l1_word, es_l1_word, de_l1_word) in tqdm(
                    desc=f'{action} for {output_scope}/{split["label"]}/{lang}',
                    iterable=inputs
                    ):
                    prompt = get_prompt(
                        prompt=args.prompt,
                        l1=lang,
                        ex_l1_words=ex_l1_words,
                        ex_l1_contexts=ex_l1_contexts,
                        ex_en_words=ex_en_words,
                        ex_difficulties=ex_difficulties,
                        l1_word=l1_word, l1_context=l1_context, en_word=en_word,
                        en_pos=en_pos,
                        all_l1_words={
                            'cn': cn_l1_word,
                            'es': es_l1_word,
                            'de': de_l1_word
                            }
                        )
                    request_id = (
                        f'bea2026st-prompting-{split["request_tag"]}-'
                        f'{lang}-{item_id}'
                        )
                    if args.batch:
                        requests_append(request_id, prompt)
                        continue

                    if args.from_batches is not None:
                        response = responses.get(request_id, None)
                        if args.rebatch_missing:
                            if response is None:
                                requests_append(request_id, prompt)
                            continue

                        if response is None:
                            batch_missing.append(request_id)
                            usage = {'batch_missing': 1}
                            text = ''
                            logprobs = []
                        else:
                            assert response['status_code'] == 200
                            r = response['body']['output'][0]['content'][0]
                            text = r['text']
                            logprobs = response_logprobs2json(r['logprobs'])
                            u = response['body']['usage']
                            usage = {} if args.dry_run else dict(
                                completion_tokens=u['output_tokens'],
                                prompt_tokens=u['input_tokens'],
                                total_tokens=u['total_tokens']
                                )
                    else:
                        if args.continue_mode:
                            output_key = f'{lang}_prompting_output'
                            logprobs_key = f'{lang}_prompting_logprobs'
                            existing = results.get(item_id, {})
                            if (
                                existing.get(output_key) is not None and
                                existing.get(logprobs_key) is not None
                                ):
                                continue
                        if args.dry_run:
                            results[item_id][f'{lang}_prompting_output'] = None
                            results[item_id][f'{lang}_prompting_logprobs'] = None
                            continue
                        rate_limit_delay = 0
                        for delay in (0, *(10 * (2 ** i) for i in range(7))):
                            delay = max(delay, rate_limit_delay)
                            rate_limit_delay = 0
                            if delay:
                                sys.stderr.write(f'Will retry in {delay} seconds.\n')
                                sleep(delay)
                            try:
                                if is_gpt52:
                                    # GPT-5.2 has special requirements, but only in the
                                    # realtime API
                                    completion = client.chat.completions.create(
                                        model=api_model,
                                        messages=[{'role': 'user', 'content': prompt}],
                                        stream=False,
                                        temperature=0,
                                        max_completion_tokens=max_tokens,  # for GPT-5.2
                                        logprobs=True,
                                        top_logprobs=5  # maxium 5 for GPT-5.2?
                                        )
                                else:
                                    completion = client.chat.completions.create(
                                        model=api_model,
                                        messages=[{'role': 'user', 'content': prompt}],
                                        stream=False,
                                        temperature=0,
                                        max_tokens=max_tokens,
                                        logprobs=True,
                                        top_logprobs=20
                                        )

                                text = completion.choices[0].message.content
                                logprobs = logprobs2json(
                                    completion.choices[0].logprobs
                                    )
                                diagnostics['system_fingerprint'] = (
                                    completion.system_fingerprint
                                    )
                            except RateLimitError as e:
                                sys.stderr.write(
                                    f'Failed request (rate limit): {e}\n'
                                    )
                                with open(
                                    logdir / 'prompting_diagnostics.json', 'a'
                                    ) as f:
                                    json.dump({'error': str(e)}, f)
                                    f.write('\n')
                                rate_limit_delay = 30
                                continue
                            except (
                                APIStatusError,
                                APITimeoutError
                                ) as e:
                                sys.stderr.write(f'Failed request: {e}\n')
                                with open(
                                    logdir / 'prompting_diagnostics.json', 'a'
                                    ) as f:
                                    json.dump({'error': str(e)}, f)
                                    f.write('\n')
                                continue
                            except Exception as e:
                                diagnostics['error'] = str(e)
                                with open(
                                    logdir / 'prompting_diagnostics.json', 'a'
                                    ) as f:
                                    json.dump(diagnostics, f)
                                    f.write('\n')
                                checkpoint_results_by_scope[output_scope] = {
                                    item_id: dict(item_results)
                                    for item_id, item_results in results.items()
                                    }
                                save_checkpoint_results(
                                    checkpoint_path,
                                    checkpoint_results_by_scope
                                    )
                                print(
                                    f'Wrote partial results to {checkpoint_path}.',
                                    file=sys.stderr
                                    )
                                print(
                                    'Continue with:\n'
                                    f'{continue_command()}',
                                    file=sys.stderr
                                    )
                                raise Exception(
                                    f'Unexpected exception during request: {e}. '
                                    'Terminating.'
                                    )
                            break
                        else:
                            checkpoint_results_by_scope[output_scope] = {
                                item_id: dict(item_results)
                                for item_id, item_results in results.items()
                                }
                            save_checkpoint_results(
                                checkpoint_path,
                                checkpoint_results_by_scope
                                )
                            print(
                                f'Wrote partial results to {checkpoint_path}.',
                                file=sys.stderr
                                )
                            print(
                                'Continue with:\n'
                                f'{continue_command()}',
                                file=sys.stderr
                                )
                            raise Exception('Exhausted retries. Terminating.')

                        usage = {} if args.dry_run else dict(
                            completion_tokens=completion.usage.completion_tokens,
                            prompt_tokens=completion.usage.prompt_tokens,
                            total_tokens=completion.usage.total_tokens
                            )

                    usage_by_lang[lang].update(usage)
                    if f'{lang}_prompting_output' in results[item_id]:
                        raise ValueError(
                            'Duplicate item_id across selected splits: '
                            f'item_id={item_id} lang={lang}'
                            )
                    results[item_id][f'{lang}_prompting_output'] = text
                    results[item_id][f'{lang}_prompting_logprobs'] = logprobs
        checkpoint_results_by_scope[output_scope] = {
            item_id: dict(item_results)
            for item_id, item_results in results.items()
            }

        if not args.batch and not args.rebatch_missing:
            df = pd.DataFrame.from_dict(results, orient='index').rename_axis(
                index=ID_COL
                )
            if not df.empty:
                df = df.sort_index()
            if not args.dry_run:
                path = prompting_output_path(outdir, output_scope, args)
                path.parent.mkdir(parents=True, exist_ok=True)
                print(f'Writing results to {path}.', file=sys.stderr)
                df.to_csv(path)
                with open(logdir / 'prompting_diagnostics.json', 'a') as f:
                    json.dump(diagnostics, f)
                    f.write('\n')

                if (
                    (not legacy_subsets_mode) and
                    (not args.final_data) and
                    args.cv_mode in ('first', 'remaining') and
                    args.limit is None and
                    args.item_ids is None and
                    (all_item_ids is not None)
                    ):
                    maybe_write_whole_from_complement(
                        args=args,
                        outdir=outdir,
                        current_scope=output_scope,
                        current_df=df,
                        all_item_ids=all_item_ids
                        )
    if batch_missing:
        print(
            f'\nWarning: Missing responses in batch, n={len(batch_missing)}:\n'
            f'{batch_missing}\n',
            file=sys.stderr
            )

    if args.batch or args.rebatch_missing:
        bpath = Path('batch')
        bpath.mkdir(parents=True, exist_ok=True)
        path = (
            bpath /
            f'requests-{scope_tag}-{args.prompt}--'
            f'{model_short_name(args.model)}{args.suffix}.jsonl'
            )
        print(f'Writing {len(requests)} requests to {path}.', file=sys.stderr)
        with open(path, 'w') as f:
            for r in requests:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        batch_id = '<dry run>'
        if not args.dry_run:
            print('Uploading requests.', file=sys.stderr)
            batch_file = client.files.create(file=open(path, 'rb'), purpose='batch')
            batch = client.batches.create(
                input_file_id=batch_file.id,
                endpoint='/v1/responses',
                completion_window='24h'
                )
            batch_id = batch.id
        print(f'Batch ID: {batch_id}')
        path = (
            bpath /
            f'batch_id-{scope_tag}-{args.prompt}--'
            f'{model_short_name(args.model)}{args.suffix}.txt'
            )
        with open(path, 'w') as f:
            f.write(batch_id)


def parse_args():
    parser = argparse.ArgumentParser()
    batch = parser.add_mutually_exclusive_group()
    batch.add_argument('--batch', action='store_true')
    batch.add_argument('--from-batches', nargs='*',
                       help='Default: last batch, optionally one or more batch IDs.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--dry-run-hardness-hist', action='store_true',
        help=(
            'Dry-run: print 10-bin per-language histogram of example hardness '
            '(abs(scaled_lr_err)) and exit without prompting.'
            )
        )
    parser.add_argument('--prompt', choices=PROMPTS, default=PROMPTS[0])
    parser.add_argument('--example-hardness', '-H',
                        choices=EXAMPLE_HARDNESS2WEIGHTS, default='easy', help=(
                            'Hardness of examples for few-shot difficulty prompts. '
                            'Using "both" doubles the number of examples. '
                            'Default: "easy" (empiricaly the best).'
                            )
                        )
    parser.add_argument('--log-examples', action='store_true',
                        help='Log few-shot examples.')
    parser.add_argument(
        '--subsets', '-s', nargs='+', choices=SUBSETS, default=None,
        help='Legacy mode: run on explicit train/dev subset(s).'
        )
    parser.add_argument(
        '--final-data', action='store_true',
        help=('Run one final split: train on full train+dev and predict on '
              'test. Output is written under predictions/prompting/test/.')
        )
    parser.add_argument('--cv-mode', choices=['whole', 'first', 'remaining'],
                        default='whole')
    parser.add_argument(
        '--splits', default=DEFAULT_SPLITS_PATH,
        help='Splits for CV mode. Default: data/cv-split-ids-5.json.'
        )
    parser.add_argument('--l1s', '-l', nargs='+', choices=L1_CODES, default=L1_CODES)

    limit = parser.add_mutually_exclusive_group()
    limit.add_argument('--limit', '-n', type=int, default=None,
                       help='Limit to first n instances. Default: no limit.')
    limit.add_argument('--item-ids', type=int, nargs='+', default=None,
                       help='Limit to specific item ids. Default: no limit.')
    limit.add_argument('--rebatch-missing', action='store_true',
                       help='Re-batch items missing in --from-batches.')
    parser.add_argument(
        '--continue', dest='continue_mode', action='store_true',
        help='Resume realtime prompting from checkpointed partial results.'
        )

    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Maximum tokens. Default: based on prompt.')
    parser.add_argument('--model', '-m', default='gpt-4.1-mini-2025-04-14',
                        help=('Use a specific model, default: gpt-4.1-mini-2025-04-14.')
                        )
    # Alternative:
    # gpt-4.1-2025-04-14 gpt-4.1-nano-2025-04-14  gpt-5.2-2025-12-11
    parser.add_argument('--output-directory', '-D', default='predictions')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-directory', '-L', default='logs')
    provider = parser.add_mutually_exclusive_group()
    provider.add_argument('--deepinfra', '--di', action='store_true')
    provider.add_argument('--deepseek', '--ds', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
