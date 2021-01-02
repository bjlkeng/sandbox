"""
Base VAE code from:
https://raw.githubusercontent.com/keras-team/keras-io/master/examples/generative/vae.py

Mixture logistic loss from previous implrementation:
https://github.com/bjlkeng/sandbox/tree/master/notebooks/pixel_cnn
"""
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    # if 'channels_last':
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, latent_dim, mixture_components=3, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.mixture_components = mixture_components
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

    @property
    def shape(self):
        return (28, 28, 1)

    def create_encoder(self):
        encoder_inputs = keras.Input(shape=self.shape)

        x = identity_block(encoder_inputs, 3, [64, 64, 256], stage=1, block='a')
        x = identity_block(encoder_inputs, 3, [64, 64, 256], stage=1, block='b')
        x = identity_block(encoder_inputs, 3, [64, 64, 256], stage=1, block='c')
        x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(self.latent_dim, activation="relu")(x)
        x = layers.Dense(self.latent_dim, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        return encoder

    def create_decoder(self):
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(7 * 7 * 256, activation="relu")(latent_inputs)
        x = layers.Reshape((7, 7, 256))(x)
        x = identity_block(x, 3, [64, 64, 256], stage=1, block='a')
        x = identity_block(x, 3, [64, 64, 256], stage=1, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=1, block='c')
        x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)

        # Create a mixture of logistic distributions , see my post on PixelCNN:
        # https://bjlkeng.github.io/posts/pixelcnn/
        # decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder_out_m = layers.Conv2DTranspose(name='x_m',
                                               filters=self.mixture_components,
                                               kernel_size=3,
                                               strides=1,
                                               padding="same")(x)
        decoder_out_invs = layers.Conv2DTranspose(name='x_invs',
                                                  filters=self.mixture_components,
                                                  kernel_size=3,
                                                  strides=1,
                                                  padding="same",
                                                  activation='softplus')(x)
        mixture_weights = layers.Conv2DTranspose(name='mixture_weights',
                                                 filters=self.mixture_components,
                                                 kernel_size=3,
                                                 strides=1,
                                                 padding="same")(x)
        decoder_outputs = layers.Concatenate()([decoder_out_m, decoder_out_invs, mixture_weights])
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        return decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Reconstruction loss using mixture of logistics
            reconstruction_loss = tf.reduce_mean(
                    pixelcnn_loss(data, reconstruction,
                                  self.shape[0], self.shape[1], self.shape[2],
                                  self.mixture_components)
            )

            # Regular KL Loss from VAE
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5

            # Putting more weight on kl_loss -- seems to converge a bit more smoothly
            total_loss = reconstruction_loss + 3 * kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def logsoftmax(x):
    ''' Numerically stable log(softmax(x)) '''
    m = K.max(x, axis=-1, keepdims=True)
    return x - m - K.log(K.sum(K.exp(x - m), axis=-1, keepdims=True))


def pixelcnn_loss(target, output, img_rows, img_cols, img_chns, n_components):
    ''' Keras PixelCNN loss function. Use a lambda to fill in the last few
        parameters

        Args:
            img_rows, img_cols, img_chns: image dimensions
            n_components: number of mixture components

        Returns:
            log-loss
    '''
    assert img_chns == 1

    # Extract out each of the mixture parameters (multiple of 3 b/c of image channels)
    output_m = output[:, :, :, :img_chns*n_components]
    output_invs = output[:, :, :, img_chns*n_components:2*img_chns*n_components]
    output_logit_weights = output[:, :, :, 2*img_chns*n_components:]

    # Repeat the target to match the number of mixture component shapes
    x = K.reshape(target, (-1, img_rows, img_cols, img_chns))
    slices = []
    for c in range(img_chns):
        slices += [x[:, :, :, c:c+1]] * n_components
    x = K.concatenate(slices, axis=-1)

    x_decoded_m = output_m
    x_decoded_invs = output_invs
    x_logit_weights = output_logit_weights

    # Pixels rescaled to be in [-1, 1] interval
    offset = 1. / 127.5 / 2.
    centered_mean = x - x_decoded_m

    cdfminus_arg = (centered_mean - offset) * K.exp(x_decoded_invs)
    cdfplus_arg = (centered_mean + offset) * K.exp(x_decoded_invs)

    cdfminus_safe = K.sigmoid(cdfminus_arg)
    cdfplus_safe = K.sigmoid(cdfplus_arg)

    # Generate the PDF (logistic) in case the `m` is way off (cdf is too small)
    # pdf = e^(-(x-m)/s) / {s(1 + e^{-(x-m)/s})^2}
    # logpdf = -(x-m)/s - log s - 2 * log(1 + e^(-(x-m)/s))
    #        = -mid_in - invs - 2 * softplus(-mid_in)
    mid_in = centered_mean * K.exp(x_decoded_invs)
    log_pdf_mid = -mid_in - x_decoded_invs - 2. * tf.nn.softplus(-mid_in)

    # Use trick from PixelCNN++ implementation to protect against edge/overflow cases
    # In extreme cases (cdfplus_safe - cdf_minus_safe < 1e-5), use the
    # log_pdf_mid and assume that density is 1 pixel width wide (1/127.5) as
    # the density: log(pdf * 1/127.5) = log(pdf) - log(127.5)
    # Add on line of best fit (see notebooks/blog post) to the difference between
    # edge case and the standard case
    edge_case = log_pdf_mid - np.log(127.5) + 2.04 * x_decoded_invs - 0.107

    # ln (sigmoid(x)) = x - ln(e^x + 1) = x - softplus(x)
    # ln (1 - sigmoid(x)) = ln(1 / (1 + e^x)) = -softplus(x)
    log_cdfplus = cdfplus_arg - tf.nn.softplus(cdfplus_arg)
    log_1minus_cdf = -tf.nn.softplus(cdfminus_arg)
    log_ll = tf.where(x <= -0.999, log_cdfplus,
                      tf.where(x >= 0.999, log_1minus_cdf,
                               tf.where(cdfplus_safe - cdfminus_safe > 1e-5,
                                        K.log(K.maximum(cdfplus_safe - cdfminus_safe, 1e-12)),
                                        edge_case)))

    # x_weights * [sigma(x+0.5...) - sigma(x-0.5 ...) ]
    # = log x_weights + log (...)
    # Compute log(softmax(.)) directly here, instead of doing 2-step to avoid overflow
    pre_result = logsoftmax(x_logit_weights) + log_ll

    result = []
    for chn in range(img_chns):
        chn_result = pre_result[:, :, :, chn*n_components:(chn+1)*n_components]
        v = tf.math.reduce_logsumexp(chn_result, axis=-1)
        result.append(v)
    result = K.batch_flatten(K.stack(result, axis=-1))

    return -K.sum(result, axis=-1)


def sigmoid(x):
    # Protect overflow
    x = np.where(x < -20, -20, x)
    x = np.where(x > 20, 20, x)

    return 1. / (1. + np.exp(-x))


def logistic_cdf(x, loc, scale):
    return sigmoid((x - loc) / scale)


def compute_pvals(m, invs):
    pvals = []
    for i in range(256):
        if i == 0:
            pval = logistic_cdf((0.5 - 127.5) / 127.5, loc=m, scale=1. / np.exp(invs))
        elif i == 255:
            pval = 1. - logistic_cdf((254.5 - 127.5) / 127.5, loc=m, scale=1. / np.exp(invs))
        else:
            pval = (logistic_cdf((i + 0.5 - 127.5) / 127.5, loc=m, scale=1. / np.exp(invs))
                    - logistic_cdf((i - 0.5 - 127.5) / 127.5, loc=m, scale=1. / np.exp(invs)))
        pvals.append(pval)

    return np.stack(pvals)


def compute_mixture(ms, invs, weights, n_comps):
    pvals = compute_pvals(ms, invs) * weights
    return np.moveaxis(np.sum(pvals, axis=-1), 0, -1)
