import numpy as np
from math import ceil, floor
from decimal import Decimal

DEBUG=False

def generate_string(size, alphabet, prob=None):
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


def code_rans(msg, alphabet, freqs, quant_bits=12, renorm_bits=16):
    '''
         msg - list of strings where each character should be in alphabet
         alphabet - list of strings representing symbols in alphabet
         freqs - List of integers f[s] s.t. p_s ~= f[s] / 2^quant_bits
         quant_bits - exponent of 2^N (quantizing factor)
         renorm_bits - n-bit renormalization
    '''
    assert len(freqs) == len(alphabet)
    assert all([type(f) == int for f in freqs])
    assert all([f > 0 for f in freqs])
    assert sum(freqs) == 1 << quant_bits
    assert quant_bits <= renorm_bits

    cdf = []
    for i in range(len(freqs) + 1):
        cdf.append(sum(freqs[:i]))

    assert len(cdf) == len(freqs) + 1

    codes = []
    code = (1 << renorm_bits) - 1
    for x in msg:
        if DEBUG:
            pcode = code
        index = alphabet.index(x)

        # Renormalization - if we would push past 2**renorm_bits, then renorm
        new_code = ((floor(code / freqs[index]) << quant_bits)
                    + (code % freqs[index])
                    + cdf[index])
        if new_code > ((1 << (2 * renorm_bits)) - 1):
            if DEBUG:
                print ('renorm')
            codes.append(code & ((1 << renorm_bits) - 1))
            code = code >> renorm_bits

        # rANS
        code = ((floor(code / freqs[index]) << quant_bits)
                + (code % freqs[index])
                + cdf[index])
        if DEBUG:
            print (pcode, ' -> ', code)

    codes.append(code)
    return codes


def decode_rans(codes, alphabet, freqs, quant_bits=8, renorm_bits=16):
    '''
         codes - coded message
         alphabet - list of strings representing symbols in alphabet
         freqs - List of integers f[s] s.t. p_s ~= f[s] / 2^quant_bits
         quant_bits - exponent of 2^N (quantizing factor)
         renorm_bits - n-bit renormalization
    '''
    assert len(freqs) == len(alphabet)
    assert all([type(f) == int for f in freqs])
    assert sum(freqs) == (1 << quant_bits)
    assert len(codes) >= 1
    assert all(c < (1 << renorm_bits) for c in codes[:-1])
    assert codes[-1] < (1 << (2*renorm_bits))

    codes = codes.copy()

    cdf = []
    for i in range(len(freqs) + 1):
        cdf.append(sum(freqs[:i]))
    assert len(cdf) == len(freqs) + 1

    msg = []
    mask = (1 << quant_bits) - 1
    code = codes.pop()
    while code >= (1 << renorm_bits):
        pcode = code
        s = code & mask
        index = np.argmax(np.array(cdf) > s) - 1
        code = (freqs[index] * (code >> quant_bits)
                + (code & mask)
                - cdf[index])
        msg = [alphabet[index]] + msg

        if (code < (1 << renorm_bits)) and codes:
            if DEBUG:
                print ('renorm')
            assert codes[-1] < (1 << renorm_bits)
            code = (code << renorm_bits) + codes.pop()

        if DEBUG:
            print (pcode, ' -> ', code)

    assert not codes, codes
    return msg
