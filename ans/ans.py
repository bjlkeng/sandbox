import numpy as np
from math import ceil, floor
from decimal import Decimal


def generate_binary_string(size, alphabet=['a', 'b'], prob=None):
    assert len(alphabet) > 0
    if prob is None:
        prob = [1 / len(alphabet)] * len(alphabet)
    assert len(alphabet) == len(prob)
    assert sum(prob) == 1

    return list(np.random.choice(alphabet, size, p=prob))


def code_uabs(msg, alphabet, prob):
    assert len(alphabet) == 2
    assert len(prob) == len(alphabet)
    assert sum(prob) == 1.0

    if prob[0] > prob[1]:
        prob[0], prob[1] = prob[1], prob[0]
        alphabet[0], alphabet[1] = alphabet[1], alphabet[0]

    p = prob[0]
    code = 1
    for x in msg:
        index = alphabet.index(x)
        # s == 1 is lower prob character
        s = 1 if index == 0 else 0

        if s == 0:
            # C(x,0) = new_x
            code = ceil(Decimal(code + 1) / (Decimal(1) - Decimal(p))) - 1
        else:
            # C(x,1) = new_x
            code = floor(Decimal(code) / Decimal(p))

    return code


def decode_uabs(code, alphabet, prob):
    assert len(alphabet) == 2
    assert len(prob) == len(alphabet)
    assert sum(prob) == 1.0

    if prob[0] > prob[1]:
        prob[0], prob[1] = prob[1], prob[0]
        alphabet[0], alphabet[1] = alphabet[1], alphabet[0]

    msg = []
    p = prob[0]
    while code > 1:
        # 0 if fract(x*p) < 1-p, else 1
        s = ceil(Decimal(code + 1) * Decimal(p)) - ceil(Decimal(code) * Decimal(p))
        if s == 0:
            # D(x) = (new_x, 0)
            code = code - ceil(Decimal(code) * Decimal(p))
        else:
            # D(x) = (new_x, 1)
            code = ceil(Decimal(code) * Decimal(p))

        msg = [alphabet[1-s]] + msg

    return msg
