import numpy as np
from tensorflow.keras import backend as K

from tensorflow.python.keras.layers import (InputSpec, Layer)
from tensorflow.python.keras.layers.normalization import BatchNormalizationBase

from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn


class Mask(Layer):
    def __init__(self, mask_type, *args, **kwargs):
        mask_type in ['check_even', 'check_odd', 'channel_even', 'channel_odd']
        self.mask_type = mask_type
        super().__init__(*args, **kwargs)

    def build_mask(self, input_shape):
        def spatial_mask_value(row, col):
            if row % 2 == 0:
                ret = 1 if col % 2 == 0 else 0
            else:
                ret = 0 if col % 2 == 0 else 1

            return ret if self.mask_type == 'check_even' else 1 - ret

        def channel_mask(chn):
            return 1 - chn % 2 if self.mask_type == 'channel_even' else chn % 2

        data = np.ones(input_shape)
        for row in range(input_shape[0]):
            for col in range(input_shape[1]):
                for chn in range(input_shape[2]):
                    if self.mask_type in ['check_even', 'check_odd']:
                        data[row, col, chn] = spatial_mask_value(row, col)
                    else:
                        assert self.mask_type in ['channel_even', 'channel_odd']
                        # channel mask
                        data[row, col, chn] = channel_mask(chn)

        return K.constant(np.ravel(data), dtype='float32', shape=input_shape)

    def build(self, input_shape):
        assert len(input_shape) == 4, \
            'Layer assumes a (batch, row, col, chn) dimensions got {}' \
            .format(input_shape)

        # Assume channel_last (tensorflow)
        channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]

        self.mask = self.build_mask(input_shape[1:])

        # Set input spec.
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        return self.mask * inputs


class FlowBatchNorm(BatchNormalizationBase):
    """ Modified BatchNorm implementation so that I can add determiniant loss
        for flow-based networks Layer
    """

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 renorm=False,
                 renorm_clipping=None,
                 renorm_momentum=0.99,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(FlowBatchNorm, self).__init__(name=name, fused=False,
                                            virtual_batch_size=None,
                                            adjustment=None,
                                            renorm=False,
                                            **kwargs)

    def call(self, inputs, training=None):
        training = self._get_training_value(training)

        assert self.virtual_batch_size is None, "Disabled"
        assert self.fused is False, "Disabled"
        assert self.adjustment is None, "Disabled"
        assert self.renorm is False, "Disabled"

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            cond = (v is not None and len(v.shape) != ndims and
                    reduction_axes != list(range(ndims - 1)))
            if cond:
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.constant_value(training)
        if training_value is False:
            mean, variance = self.moving_mean, self.moving_variance
        else:
            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
            mean, variance = self._moments(
                math_ops.cast(inputs, self._param_dtype),
                reduction_axes,
                keep_dims=keep_dims)

            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            mean = tf_utils.smart_cond(training, lambda: mean,
                                       lambda: ops.convert_to_tensor_v2(moving_mean))
            variance = tf_utils.smart_cond(
                training, lambda: variance,
                lambda: ops.convert_to_tensor_v2(moving_variance))

            new_mean, new_variance = mean, variance

            if self._support_zero_size_input():
                # Keras assumes that batch dimension is the first dimension for Batch
                # Normalization.
                input_batch_size = array_ops.shape(inputs)[0]
            else:
                input_batch_size = None

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(var, value, self.momentum,
                                                   input_batch_size)

            def mean_update():
                true_branch = lambda: _do_update(self.moving_mean, new_mean)
                false_branch = lambda: self.moving_mean
                return tf_utils.smart_cond(training, true_branch, false_branch)

            def variance_update():
                """Update the moving variance."""

                def true_branch_renorm():
                    # We apply epsilon as part of the moving_stddev to mirror the training
                    # code path.
                    moving_stddev = _do_update(self.moving_stddev,
                                               math_ops.sqrt(new_variance + self.epsilon))
                    return self._assign_new_value(
                        self.moving_variance,
                        # Apply relu in case floating point rounding causes it to go
                        # negative.
                        K.relu(moving_stddev * moving_stddev - self.epsilon))

                if self.renorm:
                    true_branch = true_branch_renorm
                else:
                    true_branch = lambda: _do_update(self.moving_variance, new_variance)

                false_branch = lambda: self.moving_variance
                return tf_utils.smart_cond(training, true_branch, false_branch)

            self.add_update(mean_update)
            self.add_update(variance_update)

        mean = math_ops.cast(mean, inputs.dtype)
        variance = math_ops.cast(variance, inputs.dtype)
        if offset is not None:
            offset = math_ops.cast(offset, inputs.dtype)
        if scale is not None:
            scale = math_ops.cast(scale, inputs.dtype)
        # TODO(reedwm): Maybe do math in float32 if given float16 inputs, if doing
        # math in float16 hurts validation accuracy of popular models like resnet.
        outputs = nn.batch_normalization(inputs,
                                         _broadcast(mean),
                                         _broadcast(variance),
                                         offset,
                                         scale,
                                         self.epsilon)
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        # bkeng: Flow loss/metric
        self.add_flow_loss(variance, scale)

        return outputs

    def add_flow_loss(self, variance, scale):
        pass


#    def call(self, inputs, training=None):
#        input_shape = K.int_shape(inputs)
#
#        assert input_shape[0] is not None, "Must explicitly specify batch size"
#        # Prepare broadcasting shape.
#        ndim = len(input_shape)
#        reduction_axes = list(range(len(input_shape)))
#        del reduction_axes[self.axis]
#        broadcast_shape = [1] * len(input_shape)
#        broadcast_shape[self.axis] = input_shape[self.axis]
#
#        # Determines whether broadcasting is needed.
#        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])
#
#        def normalize_inference():
#            if needs_broadcasting:
#                # In this case we must explicitly broadcast all parameters.
#                broadcast_moving_mean = K.reshape(self.moving_mean,
#                                                  broadcast_shape)
#                broadcast_moving_variance = K.reshape(self.moving_variance,
#                                                      broadcast_shape)
#                if self.center:
#                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
#                else:
#                    broadcast_beta = None
#                if self.scale:
#                    broadcast_gamma = K.reshape(self.gamma,
#                                                broadcast_shape)
#                else:
#                    broadcast_gamma = None
#                return K.batch_normalization(
#                    inputs,
#                    broadcast_moving_mean,
#                    broadcast_moving_variance,
#                    broadcast_beta,
#                    broadcast_gamma,
#                    axis=self.axis,
#                    epsilon=self.epsilon)
#            else:
#                return K.batch_normalization(
#                    inputs,
#                    self.moving_mean,
#                    self.moving_variance,
#                    self.beta,
#                    self.gamma,
#                    axis=self.axis,
#                    epsilon=self.epsilon)
#
#        # If the learning phase is *static* and set to inference:
#        if training in {0, False}:
#            return normalize_inference(), self.moving_mean, self.moving_variance
#
#        # If the learning is either dynamic, or set to training:
#        normed_training, mean, variance = K.normalize_batch_in_training(
#            inputs, self.gamma, self.beta, reduction_axes,
#            epsilon=self.epsilon)
#
#        # bkeng: Explicitly add determinant loss here as: `-log(gamma / sqrt(var + eps))`
#        def expand_batch(tensor):
#            return inputs * 0 + tensor
#        loss = expand_batch(-K.log(self.gamma) + 0.5 * K.log(variance + self.epsilon))
#        self.add_loss(loss, inputs=True)
#        self.add_metric(loss, aggregation='mean', name='BatchNormLoss')
#
#        if K.backend() != 'cntk':
#            sample_size = K.prod([K.shape(inputs)[axis]
#                                  for axis in reduction_axes])
#            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))
#            if K.backend() == 'tensorflow' and sample_size.dtype != 'float32':
#                sample_size = K.cast(sample_size, dtype='float32')
#
#            # sample variance - unbiased estimator of population variance
#            variance *= sample_size / (sample_size - (1.0 + self.epsilon))
#
#        self.add_update([K.moving_average_update(self.moving_mean,
#                                                 mean,
#                                                 self.momentum),
#                         K.moving_average_update(self.moving_variance,
#                                                 variance,
#                                                 self.momentum)],
#                        inputs)
#
#
#        # Pick the normalized form corresponding to the training phase.
#        return K.in_train_phase(normed_training, normalize_inference, training=training)
