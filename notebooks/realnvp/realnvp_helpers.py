import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.engine import InputSpec


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
