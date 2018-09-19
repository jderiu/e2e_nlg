from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
import numpy as np


class WordDropout(Layer):
    """Applies Word Dropout to the input.

        Word Dropout consists of setting a fraction of the indices to a dummy token. This shoudl prevent the decoder t rely soley on the language model.

        # Arguments
            rate: float between 0 and 1. Fraction of the input units to drop.
            noise_shape: 1D integer tensor representing the shape of the
                binary dropout mask that will be multiplied with the input.
                For instance, if your inputs have shape
                `(batch_size, timesteps, features)` and
                you want the dropout mask to be the same for all timesteps,
                you can use `noise_shape=(batch_size, 1, features)`.
            seed: A Python integer to use as random seed.
        """
    def __init__(self, rate, dummy_word, anneal_step=1.0, anneal_start=0, anneal_end=1000, noise_shape=None, seed=None, **kwargs):
        super(WordDropout, self).__init__(**kwargs)
        self.p = min(1., max(0., rate))
        self.dummy_word = dummy_word
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True
        self.anneal_step = anneal_step
        self.anneal_end = anneal_end
        self.anneal_start = anneal_start

    def _get_noise_shape(self, _):
        return self.noise_shape

    def call(self, inputs, training=None):
        if 0. < self.p < 1:
            anneal_weight = K.clip((self.anneal_step - self.anneal_start) / (self.anneal_end - self.anneal_start), 1e-5, 1)

            def dropped_inputs():
                shape = K.shape(inputs)
                mask = tf.where(
                    condition=K.random_uniform(shape=shape, minval=0.0, maxval=1.0, dtype='float32') <= 1 - self.p*anneal_weight,
                    x=K.ones_like(x=inputs, dtype='int32'),
                    y=K.zeros_like(x=inputs, dtype='int32')
                )
                #mask = K.random_binomial(shape=K.shape(inputs), p=1 - self.p)
                return inputs * mask + self.dummy_word * (1 - mask)

            return K.in_train_phase(dropped_inputs, inputs, training=training)

        return inputs

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(WordDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))