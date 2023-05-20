import tensorflow.python.keras.callbacks
from keras.callbacks import Callback


class CustomEarlyStopping(Callback):
    """
    Klasa służąca do wczesnego zatrzymywania uczenia modelu.
    """
    def __init__(self, threshold_acc, threshold_loss):
        """
        Konstruktor klasy CustomEarlyStopping
        :param threshold_acc: Procetowa wartość dokładności modelu
        :param threshold_loss: Procentowa wartość straty modelu
        """
        super(CustomEarlyStopping, self).__init__()
        self.threshold = threshold_acc
        self.threshold_loss = threshold_loss

    def on_epoch_end(self, epoch, logs=None):
        """
        Funkcja sprawdzająca czy model osiągnął odpowiednią dokładność i stratę.
        :param epoch: epoka z uczenia modelu
        :param logs: logi z uczenia modelu
        :return: Zatrzymuje uczenie modelu
        """
        if logs['accuracy'] > self.threshold and logs['loss'] < self.threshold_loss:
            self.model.stop_training = True