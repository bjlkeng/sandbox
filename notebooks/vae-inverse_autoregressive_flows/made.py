from random import randint

import numpy as np

from keras import backend as K
from keras.engine import Layer
from keras.layers import initializers
from keras.layers import activations
from keras.layers import regularizers
from keras.layers import constraints


class MaskingDense(Layer):
    """ Just copied code from keras Dense layer and added masking and a few other tricks:
        - Direct auto-regressive connections to output
        - Allows a second (non-autoregressive) input that is fully connected to first hidden
        - Either 1 output or 2 outputs (concatenated) that are separately
          auto-regressive wrt to the input
    """

    def __init__(self, units, out_units,
                 hidden_layers=1,
                 dropout_rate=0.0,
                 random_input_order=False,
                 activation='elu',
                 out_activation='sigmoid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 out_kernel_initializer='glorot_uniform',
                 out_bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MaskingDense, self).__init__(**kwargs)

        self.input_sel = None
        self.random_input_order = random_input_order
        self.rate = min(1., max(0., dropout_rate))
        self.kernel_sels = []
        self.units = units
        self.out_units = out_units
        self.hidden_layers = hidden_layers
        self.activation = activations.get(activation)
        self.out_activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.out_kernel_initializer = initializers.get(out_kernel_initializer)
        self.out_bias_initializer = initializers.get(out_bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def dropout_wrapper(self, inputs, training):
        if 0. < self.rate < 1.:
            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape=None, seed=None)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)

        return inputs

    def build_layer_weights(self, input_dim, units, use_bias=True, is_output=False):
        kernel_initializer = (self.kernel_initializer if not is_output
                              else self.out_kernel_initializer)
        bias_initializer = (self.bias_initializer if not is_output
                            else self.out_bias_initializer)

        kernel = self.add_weight(shape=(input_dim, units),
                                 initializer=kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        if use_bias:
            bias = self.add_weight(shape=(units,),
                                   initializer=bias_initializer,
                                   name='bias',
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        else:
            bias = None

        return kernel, bias

    def build_mask(self, shape, prev_sel, is_output):
        if is_output:
            if shape[-1] == len(self.input_sel):
                input_sel = self.input_sel
            else:
                input_sel = self.input_sel * 2
        else:
            # Disallow D-1 because it would violate auto-regressive property
            # Disallow unconnected units by sampling min from previous layer
            input_sel = [randint(np.min(prev_sel), shape[-1] - 2) for i in range(shape[-1])]

        def vals():
            in_len = len(self.input_sel)
            for x in range(shape[-2]):
                for y in range(shape[-1]):
                    if is_output:
                        yield 1 if prev_sel[x] < input_sel[y % in_len] else 0
                    else:
                        yield 1 if prev_sel[x] <= input_sel[y] else 0

        return K.constant(list(vals()), dtype='float32', shape=shape), input_sel

    def build(self, input_shape):
        if isinstance(input_shape, list):
            if len(input_shape) != 2:
                raise ValueError('Only list only supported for exactly two inputs')
            input_shape, other_input_shape = input_shape

            # Build weights for other (non-autoregressive) vector
            other_shape = (other_input_shape[-1], self.units)
            self.other_kernel, self.other_bias = self.build_layer_weights(*other_shape)

        assert len(input_shape) >= 2
        assert self.out_units == input_shape[-1] or self.out_units == 2 * input_shape[-1]

        self.kernels, self.biases = [], []
        self.kernel_masks, self.kernel_sels = [], []
        shape = (input_shape[-1], self.units)

        self.input_sel = np.arange(input_shape[-1])
        if self.random_input_order:
            np.random.shuffle(self.input_sel)
        prev_sel = self.input_sel
        for x in range(self.hidden_layers):
            # Hidden layer
            kernel, bias = self.build_layer_weights(*shape)
            self.kernels.append(kernel)
            self.biases.append(bias)

            # Hidden layer mask
            kernel_mask, kernel_sel = self.build_mask(shape, prev_sel, is_output=False)
            self.kernel_masks.append(kernel_mask)
            self.kernel_sels.append(kernel_sel)

            prev_sel = kernel_sel
            shape = (self.units, self.units)

        # Direct connection between input/output
        if self.hidden_layers > 0:
            direct_shape = (input_shape[-1], self.out_units)
            self.direct_kernel, _ = self.build_layer_weights(*direct_shape, use_bias=False,
                                                             is_output=True)
            self.direct_kernel_mask, self.direct_sel = self.build_mask(direct_shape, self.input_sel,
                                                                       is_output=True)

        # Output layer
        out_shape = (self.units, self.out_units)
        self.out_kernel, self.out_bias = self.build_layer_weights(*out_shape, is_output=True)
        self.out_kernel_mask, self.out_sel = self.build_mask(out_shape, prev_sel, is_output=True)

        self.built = True

    def call(self, inputs, training=None):
        other_input = None
        if isinstance(inputs, list):
            assert len(inputs) == 2
            assert self.hidden_layers > 0, "other input not supported if no hidden layers"
            assert hasattr(self, 'other_kernel')
            inputs, other_input = inputs

        output = inputs

        if other_input is not None:
            other = K.dot(other_input, self.other_kernel)
            other = K.bias_add(other, self.other_bias)
            other = self.activation(other)

        # Hidden layer + mask
        for i in range(self.hidden_layers):
            weight = self.kernels[i] * self.kernel_masks[i]
            output = K.dot(output, weight)

            # "other" input
            if i == 0 and other_input is not None:
                output = output + other

            output = K.bias_add(output, self.biases[i])
            output = self.activation(output)
            output = self.dropout_wrapper(output, training)

        # out_act(bias + (V dot M_v)h(x) + (A dot M_a)x + (other dot M_other)other)
        output = K.dot(output, self.out_kernel * self.out_kernel_mask)

        # Direct connection
        if self.hidden_layers > 0:
            direct = K.dot(inputs, self.direct_kernel * self.direct_kernel_mask)
            output = output + direct

        output = K.bias_add(output, self.out_bias)
        output = self.out_activation(output)
        output = self.dropout_wrapper(output, training)

        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return (input_shape[0], self.out_units)
