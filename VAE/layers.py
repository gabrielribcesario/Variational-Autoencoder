from collections.abc import Iterable
import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, units: Iterable[int], kernel_size: Iterable[Iterable[int, int]], stride: Iterable[Iterable[int, int]], 
                 padding: Iterable[str], activation: Iterable[str], latent_dim: int):
        super().__init__()

        n_conv = len(units) # Number of convolutional layers
        assert n_conv == len(kernel_size) == len(stride) == len(padding) == len(activation)

        self.conv = []
        self.ln = []
        self.act = []
        self.param_list = list(zip(units, kernel_size, stride, padding, activation))

        for u, k, s, p, a in self.param_list:
            k_init = 'he_normal' if a.lower()[-3:] == 'elu' else 'glorot_normal'
            self.conv.append(tf.keras.layers.Conv2D(u, k, s, p, kernel_initializer=k_init))
            self.ln.append(tf.keras.layers.LayerNormalization())
            self.act.append(tf.keras.layers.Activation(a))

        self.gavg = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(2 * latent_dim, kernel_initializer='he_normal', dtype=tf.float32)

    def __call__(self, x):
        for conv, ln, act in zip(self.conv, self.ln, self.act):
            x = conv(x)
            x = ln(x)
            x = act(x)
        x = self.gavg(x)
        return self.fc(x)

    def get_config(self):
        return super().get_config()
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, units: Iterable[int], kernel_size: Iterable[Iterable[int, int]], stride: Iterable[Iterable[int, int]], 
                 padding: Iterable[str], activation: Iterable[str], input_shape: Iterable[int, int, int], threshold: str = 'tanh'):
        super().__init__()

        n_conv = len(units) # Number of convolutional layers
        assert n_conv == len(kernel_size) == len(stride) == len(padding) == len(activation)

        self.ups = []
        self.conv = []
        self.ln = []
        self.act = []
        self.param_list = list(zip(units, kernel_size, stride, padding, activation))

        self.fc = tf.keras.layers.Dense(tf.math.reduce_prod(input_shape).numpy(), kernel_initializer='he_normal')
        self.fc_ln = tf.keras.layers.LayerNormalization()
        self.fc_act = tf.keras.layers.Activation('relu')
        self.reshape = tf.keras.layers.Reshape(input_shape)

        for u, k, s, p, a in self.param_list:
            k_init = 'he_normal' if a.lower()[-3:] == 'elu' else 'glorot_normal'
            self.ups.append(tf.keras.layers.UpSampling2D(s))
            self.conv.append(tf.keras.layers.Conv2D(u, k, (1,1), p, kernel_initializer=k_init))
            self.ln.append(tf.keras.layers.LayerNormalization())
            self.act.append(tf.keras.layers.Activation(a))

        # Activation: tanh or sigmoid if the reconstruction loss is cross-entropy
        self.threshold = threshold
        self.fwd_conv = tf.keras.layers.Conv2D(1, (5,5), (1,1), 'same', kernel_initializer='he_normal', activation=self.threshold, dtype=tf.float32)

    def __call__(self, x):
        x = self.fc(x)
        x = self.fc_ln(x)
        x = self.fc_act(x)
        x = self.reshape(x)
        for ups, conv, ln, act in zip(self.ups, self.conv, self.ln, self.act):
            x = ups(x)
            x = conv(x)
            x = ln(x)
            x = act(x)
        return self.fwd_conv(x)
    
    def get_config(self):
        return super().get_config()