import string
from unicodedata import normalize as nl


def process(query: str):
    query = nl('NFKC', query)
    punctuation = string.punctuation
    query = query.lower()
    for i in punctuation:
        query = query.replace(i, ' ')
    query = " ".join(query.split())
    return query


def calculate_precision(correct_sentence: str, incorrect_sentence: str, corrected_sentence: str) -> float:
    correct_sentence = process(correct_sentence)
    incorrect_sentence = process(incorrect_sentence)
    corrected_sentence = process(corrected_sentence)

    correct_words = correct_sentence.split()
    incorrect_words = incorrect_sentence.split()
    corrected_words = corrected_sentence.split()

    if corrected_words == incorrect_words:
        precision = 1

    elif len(incorrect_words) != len(correct_words):
        precision = 2
    else:
        count_true = 0
        count_false = 0
        try:
            for i in range(len(incorrect_words)):
                if corrected_words[i] == correct_words[i] and corrected_words[i] != incorrect_words[i]:
                    count_true += 1
                if corrected_words[i] != correct_words[i] and corrected_words[i] != incorrect_words[i]:
                    count_false += 1
        except:
            pass
        if count_true == 0 or count_false == 0:
            precision = 1
        else:
            precision = count_true / (count_true + count_false)

    return precision


def calculate_recall(correct_sentence: str, incorrect_sentence: str, corrected_sentence: str) -> float:
    correct_sentence = process(correct_sentence)
    incorrect_sentence = process(incorrect_sentence)
    corrected_sentence = process(corrected_sentence)

    correct_words = correct_sentence.split()
    incorrect_words = incorrect_sentence.split()
    corrected_words = corrected_sentence.split()

    if len(incorrect_words) != len(correct_words):
        recall = 2
    else:
        # false --> true
        num_corrected = 0
        try:
            for i in range(len(correct_words)):
                if incorrect_words[i] != correct_words[i] and corrected_words[i] == correct_words[i]:
                    num_corrected += 1
        except:
            pass

        # true --> false
        num_missed = 0
        try:
            for i in range(len(correct_words)):
                if incorrect_words[i] == correct_words[i] and corrected_words[i] != correct_words[i]:
                    num_missed += 1
        except:
            pass

        # false --> false
        num_error = 0
        try:
            for i in range(len(correct_words)):
                if incorrect_words[i] != correct_words[i] and corrected_words[i] != correct_words[i]:
                    num_error += 1
        except:
            pass
        # Calculate recall score
        if (num_corrected + num_missed + num_error) == 0:
            recall = 0
        else:
            recall = num_corrected / (num_corrected + num_missed + num_error)

    return recall

