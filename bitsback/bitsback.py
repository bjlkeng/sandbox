import numpy as np
from scipy.stats import norm
from scipy.special import softmax

from vae import compute_mixture
from ans import code_rans, decode_rans

BINS = {}


def norm_bins(quant_bits=16):
    ''' Cache calculation '''
    if quant_bits not in BINS:
        ppf_range = np.linspace(0., 1., (1 << quant_bits) + 1)
        BINS[quant_bits] = norm.ppf(ppf_range)
    return BINS[quant_bits]


def quantize_y_distribution(mu, log_var, sample_bits=14, quant_bits=16):
    ''' Quantizes continuous y values into bins of a standard isotropic normal '''
    assert quant_bits >= sample_bits
    # Sample equi-probable width points from the source distribution
    # (Replace -inf and + inf with closest sample to avoid weird situation)
    ppf_range = np.linspace(0., 1., (1 << sample_bits))
    ppf_range[0], ppf_range[-1] = ppf_range[1], ppf_range[-2]
    ppf_range = np.broadcast_to(ppf_range, (len(mu), len(ppf_range)))
    ppf_range = np.moveaxis(ppf_range, 0, 1)

    samples = norm.ppf(ppf_range, loc=mu, scale=np.exp(log_var))
    samples = np.moveaxis(samples, 0, 1)

    # Bucket them into the standard isotropic normal
    freqs = []
    for i in range(len(mu)):
        freq, _ = np.histogram(samples[i], bins=norm_bins(quant_bits))
        freqs.append(freq * (1 << (quant_bits - sample_bits)))
    return freqs


def decode_y(stack, mu=None, log_var=None, latent_size=50, quant_bits=16):
    assert mu is None or len(mu.shape) == 1
    assert mu is None or mu.shape == log_var.shape
    alphabet = list(range(1 << quant_bits))

    if mu is None:
        # We're trying to decode a standard isotropic normal, so freq buckets are all
        # equal-probable
        assert log_var is None
        freq = [1] * (1 << quant_bits)
        cdf = np.cumsum(freq)
        cdf = np.insert(cdf, 0, 0).astype(np.uint64)
    else:
        assert log_var is not None
        freqs = quantize_y_distribution(mu, log_var, quant_bits=quant_bits)
        cdfs = np.cumsum(freqs, axis=1)
        cdfs = np.insert(cdfs, 0, 0, axis=1).astype(np.uint64)

    # decode whatever bits on the stack interpreting them to
    # be quantizations of equal width bins of standard isotropic normal
    symbols = []
    for i in reversed(range(latent_size)):
        stack, s = decode_rans(stack, alphabet,
                               freq if mu is None else freqs[i],
                               cdf if mu is None else cdfs[i],
                               quant_bits=quant_bits)
        symbols.append(s)
    symbols = list(reversed(symbols))

    return stack, symbols


def encode_y(indexes, stack, mu=None, log_var=None, quant_bits=16):
    ''' Encode quantized y indicies for standard isotropic normals pdfs

        indexes - indexes of y into norm distributions bins
        stack - existing stack of coded symbols
    '''
    alphabet = list(range(1 << quant_bits))

    if mu is None:
        # We're trying to decode a standard isotropic normal, so freq buckets are all
        # equal-probable
        assert log_var is None
        freq = [1] * (1 << quant_bits)
        cdf = np.cumsum(freq)
        cdf = np.insert(cdf, 0, 0).astype(np.uint64)
    else:
        assert log_var is not None
        freqs = quantize_y_distribution(mu, log_var, quant_bits=quant_bits)
        cdfs = np.cumsum(freqs, axis=1)
        cdfs = np.insert(cdfs, 0, 0, axis=1).astype(np.uint64)

    for i, s in enumerate(indexes):
        stack = code_rans(s, stack, alphabet,
                          freq if mu is None else freqs[i],
                          cdf if mu is None else cdfs[i],
                          quant_bits=quant_bits)

    return stack


def quantize_pval_distribution(pvals, quant_bits=16):
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


def encode_x(data, freqs, stack, quant_bits=16):
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
        stack = code_rans(data[i], stack, alphabet, freq, cdf, quant_bits=quant_bits)

    return stack


def decode_x(freqs, stack, quant_bits=16):
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
        stack, s = decode_rans(stack, alphabet, freq, cdf, quant_bits=quant_bits)
        result.append(s)

    return stack, np.fromiter(reversed(result), np.float32)


def bitsback_append(data, stack, enc, dec, n_components=3, latent_size=50, quant_bits=16, stats=None):
    # Sample y from q(y|x_0) using the existing (ideally random) bits on the stack,
    # imaging that they were coded with a quantized q(y|x_0)
    # (This decreases coded message)
    q_dist = enc.predict((data.astype('float32') - 127.5) / 127.5)
    q_mu, q_log_var = q_dist[0], q_dist[1]
    prev_stack_size = len(stack)
    stack, indexes = decode_y(stack, q_mu[0], q_log_var[0],
                              latent_size=latent_size, quant_bits=quant_bits)
    if stats is not None:
        if 'q' not in stats:
            stats['q'] = []
        stats['q'].append(8 * 4 * (len(stack) - prev_stack_size + 1))

    # Translate quantized index to real value using bins
    bins = norm_bins(quant_bits=quant_bits)
    y = bins[indexes]

    # Extract pixel distribution from decoder, flatten and quantize to frequencies
    dist = dec.predict(y.reshape(1, -1))
    m = dist[0, :, :, :n_components]
    invs = dist[0, :, :, n_components:2*n_components]
    logit_weights = dist[0, :, :, 2*n_components:3*n_components]
    weights = softmax(logit_weights[:, :, :], axis=-1)

    pvals = compute_mixture(m, invs, weights, n_components)
    pvals = pvals.reshape(-1, pvals.shape[-1])
    freqs = quantize_pval_distribution(pvals, quant_bits=quant_bits)

    # Encode data using ANS (increase coded message size)
    data = data.reshape(-1)
    prev_stack_size = len(stack)
    stack = encode_x(data, freqs, stack)
    if stats is not None:
        if 'x' not in stats:
            stats['x'] = []
        stats['x'].append(8 * 4 * (len(stack) - prev_stack_size + 1))

    # Encode the sampled y values using isotropic normals
    prev_stack_size = len(stack)
    stack = encode_y(indexes, stack, quant_bits=quant_bits)
    if stats is not None:
        if 'y' not in stats:
            stats['y'] = []
        stats['y'].append(8 * 4 * (len(stack) - prev_stack_size + 1))

    return stack


def bitsback_pop(stack, enc, dec, shape=(28, 28, 1), n_components=3, latent_size=50, quant_bits=16):
    # Decode y according to isotropic norms (decreases coded message size)
    stack, indexes = decode_y(stack, latent_size=latent_size, quant_bits=quant_bits)

    # Translate quantized index to real value using bins
    bins = norm_bins(quant_bits=quant_bits)
    y = bins[indexes]

    # Extract pixel distribution from decoder, flatten and quantize to frequencies
    dist = dec.predict(y.reshape(1, -1))
    m = dist[0, :, :, :n_components]
    invs = dist[0, :, :, n_components:2*n_components]
    logit_weights = dist[0, :, :, 2*n_components:3*n_components]
    weights = softmax(logit_weights[:, :, :], axis=-1)

    pvals = compute_mixture(m, invs, weights, n_components)
    pvals = pvals.reshape(-1, pvals.shape[-1])
    freqs = quantize_pval_distribution(pvals, quant_bits=quant_bits)

    # Decode data using ANS (decrease coded message size)
    stack, data = decode_x(freqs, stack)

    # Get y-distribution using encoder
    q_dist = enc.predict((data.reshape(-1, shape[0], shape[1], shape[2]) - 127.5) / 127.5)
    q_mu, q_log_var = q_dist[0], q_dist[1]

    # Encode data using ANS (increase coded message size)
    stack = encode_y(indexes, stack, q_mu[0], q_log_var[0], quant_bits=quant_bits)

    return data, stack
