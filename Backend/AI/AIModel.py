import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import Sequential

from Backend.AI.CustomStops.CustomEarlyStopping import CustomEarlyStopping
from Backend.AI.Utills import prepareDataForImage


class AIModel:
    """
    Klasa reprezentująca model sieci neuronowej
    """
    def __init__(self):
        self.weights_path = 'weights-{epoch:02d}-val_accuracy_{val_accuracy:.4f}-val_loss_{val_loss:.4f}.ckpt'
        self.model_path = "model_data_v3.h5"
        self.model = None
        self.history = None


    def save_model(self):
        """
        Funkcja zapisująca model
        """
        try:
            if self.model is None:
                raise Exception("Model nie został zdefiniowany")
            print(self.model.summary())
            self.model.save(self.model_path)

        except Exception as e:
            print("blad zapisu")
            print(str(e))

    def load(self, modelPath):
        """
        Funckja wczytująca model z pliku
        :param modelPath: Ściezka do pliku modelu
        """
        try:
            self.model = tf.keras.models.load_model(modelPath)

        except Exception as e:
            print("Bład podczas ladowania modelu")
            print(str(e))
        finally:
            pass

    def modelSummary(self):
        """
        Funkcja wypisująca podsumowanie modelu
        """
        try:
            if self.model is None:
                raise Exception("Model nie został zdefiniowany")

            self.model.summary()

        except Exception:
            print("Coś poszło nie tak!")

    def buildModel(self, classesNo, load_weights = False):
        """
        Funkcja budująca model
        :param classesNo: liczba klas
        :param load_weights: czy wczytać wagi
        """
        try:
            model = Sequential()

            model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
            model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
            model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(layers.Flatten())
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(units=256, activation='relu'))
            model.add(layers.Dense(units=128, activation='relu'))
            model.add(layers.Dense(units=64, activation='relu'))
            model.add(layers.Dense(units=32, activation='relu'))
            model.add(layers.Dense(units=classesNo, activation='softmax'))
            optimizer = optimizers.rmsprop_v2.RMSprop(learning_rate=1e-5)

            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            #if exists('weights-118-val_accuracy_0.9524-val_loss_0.1476.ckpt.index') and load_weights:
                #model.load_weights('weights-118-val_accuracy_0.9524-val_loss_0.1476.ckpt.index')

            model.build((None, 224, 224, 3))

            self.model = model

        except Exception:
            print("Budowa modelu nie powiodła się")

    def trainModel(self, train_generator, valid_generator, train_size, valid_size, batch_size=1, epochs=20, save_heights = False):
        """
        Funkcja trenująca model
        :param train_generator: generator danych treningowych
        :param valid_generator: generator danych walidacyjnych
        :param train_size: rozmiar zbioru treningowego
        :param valid_size: rozmiar zbioru walidacyjnego
        :param batch_size: rozmiar batcha
        :param epochs: liczba epok
        :param save_heights: czy zapisywać wagi
        """
        try:
            if self.model is None:
                raise Exception("Model nie został zdefiniowany")

            steps_per_epoch = train_size // batch_size
            validation_steps = valid_size // batch_size

            checkpoint = ModelCheckpoint(filepath=self.weights_path,
                                         save_weights_only=True,
                                         monitor='val_loss',
                                         mode='min',
                                         save_best_only=True)

            custom_early_stopping = CustomEarlyStopping(threshold_acc=0.95, threshold_loss=1.0)

            callbacks = None
            if save_heights:
                callbacks = [checkpoint]

            self.history = self.model.fit(x=train_generator,
                                          epochs=epochs,
                                          steps_per_epoch=steps_per_epoch,
                                          validation_data=valid_generator,
                                          validation_steps=validation_steps,
                                          callbacks=[custom_early_stopping])

        except Exception as ex:
            print(str(ex))

    def predictByImagePath(self, imagePath):
        """
        Przewiduje klasę obraz na podstawie ścieżki zdjęcia
        :param imagePath: ścieżka zdjęcia
        :return: Nazwa przewidzianej klasy i procenty dopasowań do klas
        """
        try:
            if self.model is None:
                raise Exception("Model nie został zdefiniowany")
            img = prepareDataForImage(imagePath)

            if img is None or not img.any():
                raise Exception("Obraz nie znaleziony")

            img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            img_tensor = tf.image.resize(img_tensor, [224, 224])
            img_tensor = tf.expand_dims(img_tensor, 0)

            return self.__predictionLogic(img_tensor)

        except Exception as ex:
            print(str(ex))


    def predictByImage(self, image):
        """
        Przewiduje klasę obraz na podstawie przekazanego obrazu
        :param image: obraz
        :return: Nazwa przewidzianej klasy i procenty dopasowań do klas
        """
        try:
            if self.model is None:
                raise Exception("Model nie został zdefiniowany")

            return self.__predictionLogic(image)

        except Exception as ex:
            print(str(ex))

    def __predictionLogic(self, image):
        """
        Logika przewidywania klasy obrazu
        :param image: obraz
        :return: Nazwa przewidzianej klasy i procenty dopasowań do klas
        """
        try:
            if self.model is None:
                raise Exception("Model nie został zdefiniowany")

            class_names = ['Euro_10', 'Euro_100', 'Euro_20', 'Euro_200', 'Euro_5', 'Euro_50', 'Euro_500', 'Poland_10', 'Poland_100', 'Poland_20', 'Poland_200', 'Poland_50', 'Poland_500', 'UK_10', 'UK_20', 'UK_5', 'UK_50', 'USA_1', 'USA_10', 'USA_100', 'USA_2', 'USA_20', 'USA_5', 'USA_50']

            # normalizacja danych
            image = image / 255.0

            prediction = self.model.predict(x=image)

            predicted_class = np.argmax(prediction, axis=1)
            predicted_class_name = class_names[predicted_class[0]]

            return [predicted_class_name, prediction]
        except Exception as ex:
            print(str(ex))

    def plot_hist(self):
        """
        Funkcja rysująca wykres dokładności i straty modelu
        """
        try:
            if self.history is None:
                raise Exception("Histogram nie została zdefiniowany")

            hist = pd.DataFrame(self.history.history)
            hist['epoch'] = self.history.epoch

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='accuracy', mode='markers+lines'))
            fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='val_accuracy', mode='markers+lines'))
            fig.update_layout(width=1000, height=500, title='Accuracy vs. Val Accuracy', xaxis_title='Epoki',
                              yaxis_title='Accuracy', yaxis_type='linear')
            fig.show()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='loss', mode='markers+lines'))
            fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='val_loss', mode='markers+lines'))
            fig.update_layout(width=1000, height=500, title='Loss vs. Val Loss', xaxis_title='Epoki', yaxis_title='Loss',
                              yaxis_type='linear')
            fig.show()
        except Exception as ex:
            print(str(ex))


    def print_layers(self):
        """
        Funkcja wypisująca warstwy modelu
        """
        try:
            if self.model is None:
                raise Exception("Model nie został zdefiniowany")
            for layer in self.model.layers:
                print(f'layer_name: {layer.name:13} trainable: {layer.trainable}')
        except Exception as ex:
            print(str(ex))