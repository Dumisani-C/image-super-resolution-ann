import tensorflow as tf


class PixelShuffle(tf.keras.layers.Layer):
    """
    Sub-pixel convolution layer (equivalent to PyTorch's nn.PixelShuffle).
    Rearranges elements in a tensor of shape (N, H, W, C * r^2)
    into (N, H*r, W*r, C) using tf.nn.depth_to_space.
    """

    def __init__(self, scale_factor, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, x):
        return tf.nn.depth_to_space(x, self.scale_factor)

    def get_config(self):
        config = super().get_config()
        config['scale_factor'] = self.scale_factor
        return config


def build_srcnn(num_channels=3):
    """
    Super-Resolution Convolutional Neural Network (Dong et al., 2014).

    Input : Bicubic-upsampled LR image — same spatial size as HR target.
    Output: Refined SR image of identical spatial size.

    Architecture
    ------------
    1. Feature extraction : Conv2D(64, 9x9) + ReLU
    2. Non-linear mapping : Conv2D(32, 1x1) + ReLU
    3. Reconstruction     : Conv2D(3,  5x5)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=9, padding='same', activation='relu',
                               input_shape=(None, None, num_channels),
                               name='feature_extraction'),
        tf.keras.layers.Conv2D(32, kernel_size=1, padding='same', activation='relu',
                               name='non_linear_mapping'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=5, padding='same',
                               activation='sigmoid', name='reconstruction'),
    ], name='SRCNN')
    return model


def build_espcn(scale_factor=2, num_channels=3):
    """
    Efficient Sub-Pixel Convolutional Neural Network (Shi et al., 2016).

    Input : LR image at its *native* low resolution.
    Output: HR image via learned sub-pixel convolution (PixelShuffle).

    Architecture
    ------------
    1. Conv2D(64, 5x5) + ReLU
    2. Conv2D(32, 3x3) + ReLU
    3. Conv2D(C * r^2, 3x3)   where C = num_channels, r = scale_factor
    4. PixelShuffle(r)         rearranges depth -> spatial resolution
    """
    inputs = tf.keras.Input(shape=(None, None, num_channels), name='lr_input')
    x = tf.keras.layers.Conv2D(64, kernel_size=5, padding='same',
                                activation='relu', name='conv1')(inputs)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, padding='same',
                                activation='relu', name='conv2')(x)
    x = tf.keras.layers.Conv2D(num_channels * (scale_factor ** 2), kernel_size=3,
                                padding='same', name='conv3')(x)
    x = PixelShuffle(scale_factor, name='pixel_shuffle')(x)
    x = tf.keras.layers.Activation('sigmoid', name='output')(x)
    return tf.keras.Model(inputs, x, name=f'ESPCN_x{scale_factor}')


def build_model(architecture='espcn', scale_factor=2, num_channels=3):
    """Factory that returns the requested super-resolution model."""
    arch = architecture.lower()
    if arch == 'srcnn':
        return build_srcnn(num_channels=num_channels)
    elif arch == 'espcn':
        return build_espcn(scale_factor=scale_factor, num_channels=num_channels)
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose 'srcnn' or 'espcn'.")


if __name__ == '__main__':
    print('=== SRCNN ===')
    srcnn = build_model('srcnn')
    srcnn.summary()

    print('\n=== ESPCN (x2) ===')
    espcn = build_model('espcn', scale_factor=2)
    espcn.summary()
