import pandas as pd
import numpy as np
import csv
from util import download_if_necessary

CLAWS2UPOS = {
    'aj0': ['ADJ'],
    'ajc': ['ADJ'],
    'ajs': ['ADJ'],

    'av0': ['ADV'],
    'avp': ['ADV'], # NOTE: we only map to adverb (BNC uses AVP for 'adverb particle')
    'avq': ['ADV'],

    'at0': ['DET'],
    'dt0': ['DET'],
    'dtq': ['DET'],
    'dps': ['DET'],              # possessive determiner (my/your/our)

    'crd': ['NUM'],
    'ord': ['NUM'],

    'cjc': ['CCONJ'],
    'cjs': ['SCONJ'],
    'cjt': ['SCONJ'],

    'ex0': ['PRON'],             # existential 'there'
    'pnp': ['PRON'],             # personal pronoun
    'pnq': ['PRON'],             # wh-pronoun
    'pni': ['PRON'],             # indefinite pronoun
    'pnx': ['PRON'],             # reflexive pronoun
    'prp': ['ADP'],              # preposition
    'prf': ['ADP'],              # 'of' (preposition 'for/of'? in CLAWS = preposition)

    'pos': ['PART'],             # possessive marker ('s)
    'to0': ['PART'],             # infinitival 'to'

    'itj': ['INTJ'],

    'nn0': ['NOUN'],             # common noun (neutral for number / mass)
    'nn1': ['NOUN'],             # singular common noun
    'nn2': ['NOUN'],             # plural common noun
    'np0': ['PROPN'],            # proper noun

    'unc': ['X'],                # unclassified
    'xx0': ['X'],                # not otherwise classifiable / unknown
    'zz0': ['X'],                # alphabetic symbol / other

    # lexical verbs (VV*)
    'vvb': ['VERB'],
    'vvd': ['VERB'],
    'vvg': ['VERB'],
    'vvi': ['VERB'],
    'vvn': ['VERB'],
    'vvz': ['VERB'],

    # 'be' (VB*)
    'vbb': ['AUX'],
    'vbd': ['AUX'],
    'vbg': ['AUX'],
    'vbi': ['AUX'],
    'vbn': ['AUX'],
    'vbz': ['AUX'],

    # 'do' (VD*)
    'vdb': ['AUX'],
    'vdd': ['AUX'],
    'vdg': ['AUX'],
    'vdi': ['AUX'],
    'vdn': ['AUX'],
    'vdz': ['AUX'],

    # 'have' (VH*)
    'vhb': ['AUX'],
    'vhd': ['AUX'],
    'vhg': ['AUX'],
    'vhi': ['AUX'],
    'vhn': ['AUX'],
    'vhz': ['AUX'],

    # modal verb
    'vm0': ['AUX'],

    # punctuation tags (map all to PUNCT in UPOS)
    'pun': ['PUNCT'],
    'pur': ['PUNCT'],
    'puq': ['PUNCT'],

    # --- ambiguous / multi-tag BNC combos (hyphenated) ---
    'aj0-av0': ['ADJ', 'ADV'],
    'aj0-nn1': ['ADJ', 'NOUN'],
    'aj0-vvd': ['ADJ', 'VERB'],
    'aj0-vvg': ['ADJ', 'VERB'],
    'aj0-vvn': ['ADJ', 'VERB'],

    'avp-prp': ['ADV', 'PART', 'ADP'],
    'avq-cjs': ['ADV', 'SCONJ'],

    'cjs-prp': ['SCONJ', 'ADP'],
    'cjt-dt0': ['SCONJ', 'DET'],

    'crd-pni': ['NUM', 'PRON'],

    'nn1-np0': ['NOUN', 'PROPN'],
    'nn1-vvb': ['NOUN', 'VERB'],
    'nn1-vvg': ['NOUN', 'VERB'],
    'nn2-vvz': ['NOUN', 'VERB'],

    'vvd-vvn': ['VERB'],
    }

UPOS2KVL_POS = {
    'ADJ': 'adjective',
    'ADP': 'preposition',
    'ADV': 'adverb',
    'AUX': 'verb',
    'DET': 'determiner',
    'NOUN': 'noun',
    'NUM': 'number',
    'PROPN': 'noun',
    'VERB': 'verb'
    }

DEFAULT_KVL_POS = 'misc'

def claws2kvl_pos(tag: str) -> list[str]:
    upos_tags = CLAWS2UPOS.get(tag)
    if upos_tags is None:
        return [DEFAULT_KVL_POS]
    # Return a list of unique tags:
    return list({
        UPOS2KVL_POS.get(tag, DEFAULT_KVL_POS) for
        tag in upos_tags
        })


BNC_TOTAL_POS_OR_WORD = '!!WHOLE_CORPUS'  # either pos or word
TOTAL = '[TOTAL]'
FALLBACK = 0

def _read_df(written=False, spoken=False):
    assert written or spoken
    if written:
        kind = 'written' if (not spoken) else 'all'
        path = f'data/downloads/bnc_{kind}.num.gz'
        download_if_necessary(
            url=f'https://www.kilgarriff.co.uk/BNClists/{kind}.num.gz',
            path=path
            )
        df = pd.read_csv(
            path, sep=' ', names=['freq', 'word', 'pos', 'cd'],
            quoting=csv.QUOTE_NONE, na_filter=False # read 'as is'
            )
        assert (
            # all/written files differ
            (df.loc[0, 'word'] == BNC_TOTAL_POS_OR_WORD) or
            (df.loc[0, 'pos'] == BNC_TOTAL_POS_OR_WORD)
            )
        # Make same
        df.loc[0, 'word'] = TOTAL
        df.loc[0, 'pos'] = TOTAL
        return df
    else:
        assert spoken and not written
        df          = _read_df(written=True, spoken=True).set_index(['word', 'pos'])
        df_written  = _read_df(written=True).set_index(['word', 'pos'])

        df = pd.merge(
            df, df_written,
            left_index=True, right_index=True,
            how='left',
            suffixes=(None, '_written')
            ).fillna(0)
        df['freq'] -= df['freq_written']
        df['cd'] -= df['cd_written']
        df.drop(
            ['cd_written', 'freq_written'],
            axis=1, inplace=True
            )  # cannot compute cd for merged numbers
        df.reset_index(inplace=True)

    return df

class SmoothZeroType:
    pass
SmoothZero = SmoothZeroType()

class BNC:
    def __init__(
        self,
        spoken: bool = False,
        written: bool = False,
        upos: bool = False
        ):
        '''
        The optionally converted POS (upos=True) will give upper estimates in ambiguous
        cases, e.g. for the word 'late' (CLAWS POS: aj0, av0, aj0-av0, unc), the aj0-av0
        CLAWS POS will contribute to both 'adjective' and 'adverb' UPOS. Note that we do not
        do this with 'unc' (unclassified), which is typically a relatively small share.
        (TODO?)

        This means that after conversion the sum of all POS (indexed by [word, FALLBACK]) is
        not equal to the sum of all *converted* POS!

        '''

        df = _read_df(written=written, spoken=spoken)
        assert df.loc[0, 'word'] == TOTAL

        total_tokens = df.loc[0, 'freq']
        total_types = len(df) - 1

        df_grouped = df[['word', 'freq']].groupby(['word'])
        freq_sum_pos = df_grouped.sum() + df_grouped.count()  # smoothing
        freq_sum_pos.index = pd.MultiIndex.from_product(
            [freq_sum_pos.index, [FALLBACK]],
            names=['word', 'pos'],
            )

        if upos:
            # Note: we can sum frequencies, but not cd
            df = df.rename(columns={'pos': 'claws_pos'})
            df['pos'] = df['claws_pos'].map(claws2kvl_pos)
            df = df.explode('pos', ignore_index=True)
            df_grouped = df[['word', 'pos', 'freq']].groupby(['word', 'pos'])
            df = df_grouped.sum() + df_grouped.count()  # smoothing
        else:
            df = df.set_index(['word', 'pos'])
            df['freq'] += 1  # smoothing

        df = pd.concat([df, freq_sum_pos])

        df.loc[(FALLBACK, FALLBACK), 'freq'] = 1   # smoothing (zero)

        df['freq'] = np.log10(df['freq'] / (total_tokens + total_types)) # smoothing

        self.df = df

    @property
    def smooth_zero(self) -> float:
        return self.df.loc[(FALLBACK, FALLBACK), 'freq']

    # The following methods are slow for large numbers of lookups, better to use
    # the dataframe directly.

    def log_frequency(self, word, pos, fill_value=SmoothZero, pos_fill=False):
        if not word in self.df.index:
            return (
                fill_value if (fill_value is not SmoothZero) else
                self.df.loc[(FALLBACK, FALLBACK), 'freq']
                )

        word_freqs = self.df.loc[word]

        if pos not in word_freqs.index:
            if not pos_fill:
                return (
                    fill_value if (fill_value is not SmoothZero) else
                    self.df.loc[(FALLBACK, FALLBACK), 'freq']
                    )
            return word_freqs.loc[FALLBACK, 'freq']    # TODO max instead of sum?

        return word_freqs.loc[pos, 'freq']


    def log_frequency_any_pos(self, word, fill_value=SmoothZero):
        if not (word, FALLBACK) in self.df.index:
            return (
                fill_value if (fill_value is not SmoothZero) else
                self.df.loc[(FALLBACK, FALLBACK), 'freq']
                )
        return self.df.loc[(word, FALLBACK), 'freq']


    def __contains__(self, word):
        return (word, FALLBACK) in self.df.index


# if __name__ == '__main__':
#     get_bnc()

# def get_bnc_spoken_written_freq_data(
#     spoken: bool = False,
#     written: bool = False
#     ) -> FrequencyData:
#
#     assert written or spoken    # at least one of them
#
#     if spoken:
#         fd  = FrequencyData.from_file_url(
#             url='https://www.kilgarriff.co.uk/BNClists/all.num.gz',
#             filename='data/downloads/bnc_all.num.gz',
#             header=['freq', 'word', 'pos', 'cd'], cols=['word', 'freq', 'cd'],
#             delimiter=' ', total_row='!!WHOLE_CORPUS'
#             )
#         if written:
#             return fd
#
#     assert written != spoken  # either spoken or written
#
#     fdw = FrequencyData.from_file_url(
#         url='https://www.kilgarriff.co.uk/BNClists/written.num.gz',
#         filename='data/downloads/bnc_written.num.gz',
#         header=['freq', 'word', 'pos', 'cd'], cols=['word', 'freq', 'cd'],
#         delimiter=' ', total_row='!!ANY'
#         )
#     if written:
#         return fdw
#
#     assert spoken and not written
#
#     return fd.difference(fdw)
