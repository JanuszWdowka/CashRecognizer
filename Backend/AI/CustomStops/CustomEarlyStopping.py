import tensorflow.python.keras.callbacks
from keras.callbacks import Callback


class CustomEarlyStopping(Callback):
    def __init__(self, threshold_acc, threshold_loss):
        super(CustomEarlyStopping, self).__init__()
        self.threshold = threshold_acc
        self.threshold_loss = threshold_loss

    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] > self.threshold and logs['loss'] < self.threshold_loss:
            self.model.stop_training = True