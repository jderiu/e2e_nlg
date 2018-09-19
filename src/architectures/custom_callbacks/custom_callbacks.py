import keras.backend as K
import numpy as np
from keras.callbacks import Callback


class StepCallback(Callback):
    def __init__(self, alpha, steps_per_epoch):
        self.alpha = alpha
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        super(StepCallback, self).__init__()

    def on_batch_end(self, batch, logs=None):
        value = self.steps_per_epoch*self.current_epoch + batch
        K.set_value(self.alpha, value)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch


class TrainingCallback(Callback):

    def __init__(self, training):
        self.training = training
        super(TrainingCallback, self).__init__()

    def on_batch_end(self, batch, logs=None):
        K.set_value(self.training, False)

    def on_batch_begin(self, batch, logs=None):
        K.set_value(self.training, True)


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered."""

    def __init__(self):
        self.terminated_on_nan = False
        super(TerminateOnNaN, self).__init__()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True
                self.terminated_on_nan = True


def pretrain_discriminator(model, data, vocab):
    nsamples = data.shape[0]
    sample_size = data.shape[1]
    y_orig = np.ones((nsamples, ))
    y_fake = np.zeros((nsamples, ))

    fake_data = np.random.randint(low=0, high=max(vocab.values()), size=(nsamples, sample_size))
    sen_lens = np.random.normal(loc=60, scale=20, size=nsamples)


    train_set = np.vstack((data, fake_data))
    labels = np.vstack((y_orig, y_fake))

    model.fit(train_set, labels, epochs=20)


class MultiModelCheckpoint(Callback):
    def __init__(self, models, period=1):
        super(MultiModelCheckpoint, self).__init__()
        self.models = models
        self.period = period
        self.epochs_since_last_save = 0


    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            for model, filepath in self.models:
                model.save_weights(filepath, overwrite=True)
