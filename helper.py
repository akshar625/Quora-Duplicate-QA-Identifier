# helper.py

import numpy as np

def query_point_creator(q1, q2, cv):
    input_query = []

    q1_len = len(q1)
    q2_len = len(q2)
    q1_num_words = len(q1.split(" "))
    q2_num_words = len(q2.split(" "))

    input_query.append(q1_len)
    input_query.append(q2_len)
    input_query.append(q1_num_words)
    input_query.append(q2_num_words)

    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    word_common = len(w1 & w2)
    word_total = len(w1) + len(w2)
    word_share = round(word_common / word_total, 2) if word_total != 0 else 0

    input_query.append(word_common)
    input_query.append(word_total)
    input_query.append(word_share)

    q1_bow = cv.transform([q1]).toarray()

    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 7), q1_bow, q2_bow))
