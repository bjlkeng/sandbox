import numpy as np
from math import floor


DEBUG = False


def code_rans(symbol, stack, alphabet, freqs, cdf=None, quant_bits=16, renorm_bits=32):
    '''
         Returns stack as a list where each element is renorm_bits long except
         for the last one which is 2*renorm_bits long

         symbol - string corresponding to a symbol alphabet
         stack - stack of codes from previous encoded symbols
         alphabet - list of strings representing symbols in alphabet
         freqs - List/nd.array of integers f[s] s.t. p_s ~= f[s] / 2^quant_bits
         cdf - np.array of cumulative sum of frequencies, if None will be calculated from freqs
         quant_bits - exponent of 2^N (quantizing factor)
         renorm_bits - n-bit renormalization
    '''
    if DEBUG:
        assert len(freqs) == len(alphabet)
        assert all([f > 0 for f in freqs])
        assert sum(freqs) == 1 << quant_bits
        assert type(stack) == list
        assert quant_bits <= renorm_bits

    if cdf is None:
        cdf = np.cumsum(freqs)
        cdf = np.insert(cdf, 0, 0).astype(np.uint64)
        assert len(cdf) == len(freqs) + 1

    if not stack:
        codes = []
        code = 1
    else:
        codes = stack
        code = int(codes.pop())

    if DEBUG:
        pcode = code
    index = alphabet.index(symbol)
    assert int(freqs[index]) != 0, 'Symbol has zero probability - index = %d' % index

    # Renormalization - if we would push past 2**renorm_bits, then renorm
    new_code = ((floor(code // int(freqs[index])) << quant_bits)
                + (code % int(freqs[index]))
                + int(cdf[index]))
    if new_code > ((1 << (2 * renorm_bits)) - 1):
        if DEBUG:
            print('renorm')
        codes.append(code & ((1 << renorm_bits) - 1))
        assert codes[-1] <= (1 << renorm_bits) - 1
        code = code >> renorm_bits

    # rANS
    code = ((floor(code // int(freqs[index])) << quant_bits)
            + (code % int(freqs[index]))
            + int(cdf[index]))

    if DEBUG:
        print(pcode, '(', type(pcode), ')', ' -> ', code, ' ', type(code))

    assert type(code) == int

    codes.append(code)
    return codes


def decode_rans(stack, alphabet, freqs, cdf=None, quant_bits=16, renorm_bits=32):
    '''
         stack - stack of coded message (see return of above function)
         alphabet - list of strings representing symbols in alphabet
         freqs - List/nd.array of integers f[s] s.t. p_s ~= f[s] / 2^quant_bits
         cdf - np.array of cumulative sum of frequencies, if None will be calculated from freqs
         quant_bits - exponent of 2^N (quantizing factor)
         renorm_bits - n-bit renormalization
    '''
    if DEBUG:
        assert len(freqs) == len(alphabet)
        assert sum(freqs) == (1 << quant_bits)
        assert len(stack) >= 1
        assert all(c < (1 << renorm_bits) for c in stack[:-1])
        assert stack[-1] < (1 << (2*renorm_bits))

    codes = stack

    if cdf is None:
        cdf = np.cumsum(freqs)
        cdf = np.insert(cdf, 0, 0).astype(int)
        assert len(cdf) == len(freqs) + 1

    mask = (1 << quant_bits) - 1
    pcode, plen = codes[-1], len(codes)
    code = int(codes.pop())
    if code >= 1:
        if DEBUG:
            pcode = code
        s = code & mask
        index = np.argmax(cdf > s) - 1
        code = (int(freqs[index]) * (code >> quant_bits)
                + (code & mask)
                - int(cdf[index]))

        symbol = alphabet[index]
        if (code < (1 << renorm_bits)) and codes:
            if DEBUG:
                print('renorm')
            assert codes[-1] < (1 << renorm_bits)
            code = (code << renorm_bits) + codes.pop()
        codes.append(code)

        if DEBUG:
            print(pcode, ' -> ', code)

        assert (0 < codes[-1] < pcode or len(codes) < plen
                or (pcode == codes[-1] and freqs[index] == (1 << quant_bits))), \
            (pcode, codes[-1], plen, len(codes), index, freqs[index], cdf[index])
        return codes, symbol
    else:
        return [], None
