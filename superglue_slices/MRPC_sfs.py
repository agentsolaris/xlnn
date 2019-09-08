from snorkel.slicing.sf import slicing_function
#from .general_sfs import slice_func_dict as general_slice_func_dict

@slicing_function()
def slice_temporal_preposition(example):
    temporal_prepositions = ["after", "before", "past"]
    both_sentences = example['#1 String'] + example['#2 String']
    return any([p in both_sentences for p in temporal_prepositions])


@slicing_function()
def slice_possessive_preposition(example):
    possessive_prepositions = ["inside of", "with", "within"]
    both_sentences = example['#1 String'] + example['#2 String']
    return any([p in both_sentences for p in possessive_prepositions])


@slicing_function()
def slice_is_comparative(example):
    comparative_words = ["more", "less", "better", "worse", "bigger", "smaller"]
    both_sentences = example['#1 String'] + example['#2 String']
    return any([p in both_sentences for p in comparative_words])


@slicing_function()
def slice_is_quantification(example):
    quantification_words = ["all", "some", "none"]
    both_sentences = example['#1 String'] + example['#2 String']
    return any([p in both_sentences for p in quantification_words])


@slicing_function()
def slice_short_hypothesis(example, thresh=5):
    return len(example['#2 String'].split()) < thresh


@slicing_function()
def slice_long_hypothesis(example, thresh=15):
    return len(example['#2 String'].split()) > thresh


@slicing_function()
def slice_short_premise(example, thresh=10):
    return len(example['#1 String'].split()) < thresh


@slicing_function()
def slice_long_premise(example, thresh=100):
    return len(example['#1 String'].split()) > thresh


@slicing_function()
def slice_where(example):
    sentences = example['#1 String'] + example['#2 String']
    return "where" in sentences


@slicing_function()
def slice_who(example):
    sentences = example['#1 String'] + example['#2 String']
    return "who" in sentences


@slicing_function()
def slice_what(example):
    sentences = example['#1 String'] + example['#2 String']
    return "what" in sentences


@slicing_function()
def slice_when(example):
    sentences = example['#1 String'] + example['#2 String']
    return "when" in sentences


@slicing_function()
def slice_and(example):
    sentences = example['#1 String'] + example['#2 String']
    return "and" in sentences


@slicing_function()
def slice_but(example):
    sentences = example['#1 String'] + example['#2 String']
    return "but" in sentences


@slicing_function()
def slice_or(example):
    sentences = example['#1 String'] + example['#2 String']
    return "or" in sentences


@slicing_function()
def slice_multiple_articles(example):
    sentences = example['#1 String'] + example['#2 String']
    multiple_indefinite = (
        sum([int(x == "a") for x in sentences.split()]) > 1
        or sum([int(x == "an") for x in sentences.split()]) > 1
    )
    multiple_definite = sum([int(x == "the") for x in sentences.split()]) > 1
    return multiple_indefinite or multiple_definite

@slicing_function()
def slice_not(example):
    sentences = example['#1 String'] + example['#2 String']
    return "not" in sentences

@slicing_function()
def slice_uncertain(example):
    uncertain_words = ["doubtful", "undecided", "unsure", "indefinite"]
    both_sentences = example['#1 String'] + example['#2 String']
    return any([p in both_sentences for p in uncertain_words])

@slicing_function()
def slice_freq(example):
    freq_words = ["hourly", "annually", "usually", "often", "daily"]
    both_sentences = example['#1 String'] + example['#2 String']
    return any([p in both_sentences for p in freq_words])

@slicing_function()
def slice_speech(example):
    speech_words = ["\" said", "\" asked", "\" replied"]
    both_sentences = example['#1 String'] + example['#2 String']
    return any([p in both_sentences for p in speech_words])

slices = [
    slice_not,
    slice_uncertain,
    slice_freq,
    slice_speech,
    slice_temporal_preposition,
    slice_possessive_preposition,
    slice_is_comparative,
    slice_is_quantification,
    slice_short_hypothesis,
    slice_long_hypothesis,
    slice_short_premise,
    slice_long_premise,
    slice_where,
    slice_who,
    slice_what,
    slice_when,
    slice_and,
    slice_or,
    slice_but,
    slice_multiple_articles,
]


slice_func_dict = {slice.name: slice for slice in slices}
#slice_func_dict.update(general_slice_func_dict)

