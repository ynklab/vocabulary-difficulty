from finetune_llm_spaces import PointScale
from kvl import spaced_clue


LANG2NAME = {'cn': 'Chinese', 'de': 'German', 'es': 'Spanish'}

PROMPTS = [
    # Best performances, r >= .86
    'no_pos_rev', 'no_pos_l1_rev',  # .8652 .8648
    'no_pos_hash',                  # .861
    'pos_rev', '',                  # .860
    'test_clue',                    # .863
    'how_many',

    # Lower performance:
    'no_pos',               # .852
    'no_pos_rev_hash',      # .835
    'no_pos_no_clue_rev',   # .839
    'test',                 # .852 (notably without clue)
    'test_clue_rev_hash',   # .850
    'test_clue_rev',        # .848
    'test_native',          # .846
    'bare_difficulty',      # .829
    'first_try',            # .857
    'ft1', 'ft2'            # .838, .849
    ]


_PROMPTS_FOR_FACILITY = {'how_many'}


def prompt_is_for_difficulty(prompt: str) -> bool:
    if prompt not in PROMPTS:
        raise ValueError(f'Unknown prompt: {prompt}')
    return prompt not in _PROMPTS_FOR_FACILITY


def _difficulty_scale_text(scale: PointScale):
    return (
        f'from {scale.min} to {scale.max} '
        f'({scale.min}=very easy, {scale.max}=very difficult)'
        )


def build_prompt(l1, row, prompt, scale: PointScale):
    l1_name = LANG2NAME[l1]
    l1_word = row[f'{l1}_L1_source_word']
    l1_context = row[f'{l1}_L1_context']
    en_word = row['en_target_word']
    clue = spaced_clue(en_word)
    pos = row['en_target_pos']

    match prompt:
        case 'no_pos':
            return (
                f'{l1_name} word: {l1_word}\n'
                f'{l1_name} context: {l1_context}\n'
                f'Clue: {clue}\n'
                f'English word: {en_word}\n'
                f'Rate how difficult it is for learners to guess the English '
                f'word based on the {l1_name} word, context and clue on a scale '
                f'{_difficulty_scale_text(scale)}:'
                )
        case 'no_pos_hash':
            return (
                f'### {l1_name} word: {l1_word}\n'
                f'### {l1_name} context: {l1_context}\n'
                f'### Clue: {clue}\n'
                f'### English word: {en_word}\n'
                f'### Rate how difficult it is for learners to guess the English '
                f'word based on the {l1_name} word, context and clue on a scale '
                f'{_difficulty_scale_text(scale)}:'
                )
        case 'no_pos_rev_hash':
            return (
                f'### Rate how difficult it is for learners to guess the English '
                f'word based on the {l1_name} word, context and clue on a scale '
                f'{_difficulty_scale_text(scale)}.\n'
                f'### {l1_name} word: {l1_word}\n'
                f'### {l1_name} context: {l1_context}\n'
                f'### Clue: {clue}\n'
                f'### English word: {en_word}\n'
                f'### Difficulty:'
                )
        case 'how_many':
            if scale.min != 0:
                raise ValueError(
                    f'Incompatible scale minimum. Required: 0. Got: {scale.min}.'
                    )
            return (
                f'Estimate how many out of {scale.max} learners would correctly guess '
                f'the English word based on the {l1_name} word, context and clue.\n'
                f'{l1_name} word: {l1_word}\n'
                f'{l1_name} context: {l1_context}\n'
                f'Clue: {clue}\n'
                f'English word: {en_word}\n'
                f'Estimate of correct guesses:'
                )
        case 'no_pos_rev':
            return (
                f'Rate how difficult it is for learners to guess the English '
                f'word based on the {l1_name} word, context and clue on a scale '
                f'{_difficulty_scale_text(scale)}.\n'
                f'{l1_name} word: {l1_word}\n'
                f'{l1_name} context: {l1_context}\n'
                f'Clue: {clue}\n'
                f'English word: {en_word}\n'
                f'Difficulty:'
                )
        case 'pos_rev':
            return (
                f'Rate how difficult it is for learners to guess the English '
                f'word based on the {l1_name} word, context and clue on a scale '
                f'{_difficulty_scale_text(scale)}.\n'
                f'{l1_name} word: {l1_word}\n'
                f'{l1_name} context: {l1_context}\n'
                f'Clue: {clue}\n'
                f'English word: {en_word}\n'
                f'Part of speech: {pos}\n'
                f'Difficulty:'
                )
        case 'no_pos_l1_rev':
            return (
                f'Rate how difficult it is for {l1_name} L1 English learners to guess '
                f'the English word based on the {l1_name} word, context and clue on a '
                f'scale {_difficulty_scale_text(scale)}.\n'
                f'{l1_name} word: {l1_word}\n'
                f'{l1_name} context: {l1_context}\n'
                f'Clue: {clue}\n'
                f'English word: {en_word}\n'
                f'Difficulty:'
                )
        case 'no_pos_no_clue_rev':
            return (
                f'Rate how difficult it is for learners to guess the English '
                f'word based on the {l1_name} word and context on a scale '
                f'{_difficulty_scale_text(scale)}.\n'
                f'{l1_name} word: {l1_word}\n'
                f'{l1_name} context: {l1_context}\n'
                f'English word: {en_word}\n'
                f'Difficulty:'
                )
        case 'ft2':
            # Note: No clue, no pos, "L1 context":
            return (
                f'{l1_name} word: {l1_word}\n'
                f'{l1_name} context: {l1_context}\n'
                f'English word: {en_word}\n'
                f'Rate how difficult it is for learners to guess the English '
                f'word based on the {l1_name} word and context on a scale '
                f'{_difficulty_scale_text(scale)}:'
                )
        case 'ft1':
            # Note: No clue, no pos:
            return (
                f'{l1_name} word: {l1_word}\n'
                f'Context: {l1_context}\n'
                f'English word: {en_word}\n'
                f'Rate how difficult it is for learners to guess the English '
                f'word based on the {l1_name} word and context on a scale '
                f'{_difficulty_scale_text(scale)}:'
                )
        case 'test':
            return (
                f'Vocabulary test item for L1={l1_name}, L2=English:\n'
                f'L1 word: {l1_word}\n'
                f'L1 context: {l1_context}\n'
                f'Correct answer: {en_word}\n'
                f'Test item difficulty ({scale.min} to {scale.max}):'
                )
        case 'test_native':
            return (
                f'English vocabulary test item::\n'
                f'L1 word: {l1_word}\n'
                f'L1 context: {l1_context}\n'
                f'Correct answer: {en_word}\n'
                f'Test item difficulty ({scale.min} to {scale.max}) '
                f'for native {l1_name} learners:'
                )
        case 'test_clue_rev':
            return (
                f'Rate vocabulary test item difficulty '
                f'from {scale.min} to {scale.max}.\n'
                f'L1={l1_name}, L2=English\n'
                f'L1 word: {l1_word}\n'
                f'L1 context: {l1_context}\n'
                f'Clue: {clue}\n'
                f'Correct answer: {en_word}\n'
                f'Difficulty:'
                )
        case 'test_clue_rev_hash':
            return (
                f'### Rate vocabulary test item difficulty '
                f'from {scale.min} to {scale.max}.\n'
                f'### L1={l1_name}, L2=English\n'
                f'### L1 word: {l1_word}\n'
                f'### L1 context: {l1_context}\n'
                f'### Clue: {clue}\n'
                f'### Correct answer: {en_word}\n'
                f'### Difficulty:'
                )
        case 'test_clue':
            return (
                f'Vocabulary test item for L1={l1_name}, L2=English:\n'
                f'L1 word: {l1_word}\n'
                f'L1 context: {l1_context}\n'
                f'Clue: {clue}\n'
                f'Correct answer: {en_word}\n'
                f'Test item difficulty ({scale.min} to {scale.max}):'
                )
        case 'bare_difficulty':
            return (
                f'{l1_word} ### '
                f'{l1_context} ### '
                f'{clue} ### '
                f'{en_word} ### '
                f'Difficulty ({scale.min} to {scale.max}):'
                )
        case 'first_try':
            # Note: No clue
            return (
                f'{l1_name} word: {l1_word}\n'
                f'Context: {l1_context}\n'
                f'Part of Speech: {pos}\n'
                f'English word: {en_word}\n'
                f'Rate how difficult it is for learners to guess the English '
                f'word based on the {l1_name} word and context on a scale '
                f'{_difficulty_scale_text(scale)}:'
                )

    raise ValueError(f'Unimplemented prompt: {prompt}')
