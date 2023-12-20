import math
import os
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def count_unique(dict_spam: dict, dict_not_spam: dict) -> int:
    """
    Count unique keys in two dicts
    :param dict_spam:
    :param dict_not_spam:
    :return:
    """
    count = len(dict_spam) + len(dict_not_spam)
    for k in dict_spam.keys():
        if k in dict_not_spam.keys():
            count -= 1

    return count


def scan_files(dir_name: str) -> (int, dict):
    """
    Scan files from the directory and read text into dict.
    :param dir_name: Name of the dir
    :return: (int, dict)
    """
    dict_file = {}
    DIR = BASE_DIR / dir_name
    len_dir = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    for i in range(1, len_dir+1):
        file_name = BASE_DIR / dir_name / f'{i}.txt'
        file = open(file_name)
        file_text = file.readline().split()
        for word in file_text:
            dict_file.__setitem__(word, 0)
            dict_file[word] += 1
        file.close()

    return len_dir, dict_file


def naive_bayes_classifier(dict_text: dict) -> None:
    """
    Realisation of Naive Bayes Classifier
    :param dict_text: Text file with message for analyze
    :return: A message - spam or not
    """
    len_spam, spam_words = scan_files("Spam")
    len_not_spam, not_spam_words = scan_files("Not_Spam")
    len_dirs = [len_spam, len_not_spam]
    clusters = [0, 0]
    unique_words = count_unique(spam_words, not_spam_words)
    count_l = [sum(spam_words.values()), sum(not_spam_words.values())]
    count_w = np.zeros((2, unique_words))

    for k in dict_text.keys():
        i = 0
        if k in spam_words.keys():
            count_w[0][i] = spam_words[k]

        if k in not_spam_words.keys():
            count_w[1][i] = not_spam_words[k]

    for i in range(len(clusters)):
        clusters[i] = len_dirs[i] / unique_words
        for j in range(len(count_w[0])):
            clusters[i] += math.log((count_w[i][j] + 1)/(unique_words + count_l[i]))

    result_value = max(clusters)
    result_index = clusters.index(result_value)
    if result_index == 0:
        print("Данное сообщение - спам")
    else:
        print("Данное сообщение - не спам")


if __name__ == "__main__":
    _, test = scan_files("text_test")
    naive_bayes_classifier(test)




