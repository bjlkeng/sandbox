import numpy as np
from scipy.stats import norm
from ans import code_rans, decode_rans


def quantize_y(y, quant_bits=16):
    ''' Quantizes y by returning index of bin corresponding the pdf of isotropic normals
        into equal probability buckets '''
    ppf_range = np.linspace(0., 1., (1 << quant_bits) + 1)
    bins = norm.ppf(ppf_range, np.zeros(len(ppf_range)), np.ones(len(ppf_range)))
    return np.searchsorted(bins, y)


def dequantize_y(quant_y, quant_bits=16):
    ''' Dequantizes y translating index of bin into quantized real values '''
    ppf_range = np.linspace(0., 1., (1 << quant_bits) + 1)
    bins = norm.ppf(ppf_range, np.zeros(len(ppf_range)), np.ones(len(ppf_range)))
    return bins[quant_y]


def decode_y(stack, mu, log_var, quant_bits=16):
    assert len(mu.shape) == 1
    assert mu.shape == log_var.shape
    alphabet = list(range(1 << quant_bits))
    freqs = [1] * (1 << quant_bits)
    cdf = np.cumsum(freqs)
    cdf = np.insert(cdf, 0, 0).astype(np.uint64)
    locs = mu
    scales = np.exp(log_var)

    # decode whatever bits on the stack interpreting them to
    # be quantizations of equal width bins of N(mu, log_var) pdf
    symbols = []
    for i in range(len(locs)):
        stack, s = decode_rans(stack, alphabet, freqs, cdf)
        symbols.append(s)
    symbols = list(reversed(symbols))

    # Convert quantization index to actual continuous value
    y = norm.ppf(np.array(symbols) * 1. / (1 << quant_bits), locs, scales)
    return stack, y


def encode_y(quant_y, stack, quant_bits=16):
    ''' Encode quantized y indicies for isotropic normals pdfs

        quant_y - indexes of y into isotropic norm bins via quantize_y()
        stack - existing stack of coded symbols
    '''
    assert len(quant_y.shape) == 1

    alphabet = list(range(1 << quant_bits))
    freqs = [1] * (1 << quant_bits)
    cdf = np.cumsum(freqs)
    cdf = np.insert(cdf, 0, 0).astype(np.uint64)

    for s in quant_y:
        # code y symbols
        stack = code_rans(s, stack, alphabet, freqs, cdf)

    return stack


def quantize_distribution(pvals, quant_bits=16):
    ''' Translate probability distribution into a frequency distribution
        ensuring each bucket has at least one count
        pvals - N x 256 numpy array of 256 pixel values
    '''
    # Add +1 to each zero bin to ensure non-zero probability
    freqs = np.round(pvals * (1 << quant_bits))
    freqs = np.where(freqs == 0, freqs + 1, freqs)

    # Re-adjust frequencies so that it adds up to exactly 2^quant_bits
    # (Shave it off from largest bin)
    adjustment = (1 << quant_bits) - np.sum(freqs, axis=-1)
    maxval_index = np.argmax(freqs, axis=-1)
    for i in range(len(adjustment)):
        freqs[i, maxval_index[i]] += adjustment[i]

    return freqs


def encode_x(data, freqs, stack):
    ''' Encodes the (flattened) image using ANS

        data - flattened data
        freqs - frequencies for the flattened data
        stack - existing stack of coded symbols
    '''
    assert len(data) == freqs.shape[0], (data.shape, freqs.shape)
    for i in range(len(data)):
        freq = freqs[i]
        alphabet = list(range(len(freq)))
        cdf = np.cumsum(freq)
        cdf = np.insert(cdf, 0, 0).astype(np.uint64)
        stack = code_rans(data[i], stack, alphabet, freq, cdf)

    return stack


def decode_x(freqs, stack):
    ''' Decodes the flattened image using ANS

        freqs - frequencies for the flattened data
        stack - existing stack of coded symbols
    '''
    result = []
    for i in reversed(range(freqs.shape[0])):
        freq = freqs[i]
        alphabet = list(range(len(freq)))
        cdf = np.cumsum(freq)
        cdf = np.insert(cdf, 0, 0).astype(np.uint64)
        stack, s = decode_rans(stack, alphabet, freq, cdf)
        result.append(s)

    return stack, list(reversed(result))
