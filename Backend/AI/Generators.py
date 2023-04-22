import os
import shutil
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Sequential


class Generator:
    def __init__(self):
        self.datagen = self.__create_datagen()

    def __create_datagen(self):
        train_datagen = ImageDataGenerator(
            rotation_range=10,  # zakres kąta o który losowo zostanie wykonany obrót obrazów
            rescale=1. / 255.,
            width_shift_range=0.2,  # pionowe przekształcenia obrazu
            height_shift_range=0.2,  # poziome przekształcenia obrazu
            shear_range=0.2,  # zares losowego przycianania obrazu
            zoom_range=0.2,  # zakres losowego przybliżania obrazu
            horizontal_flip=True,  # losowe odbicie połowy obrazu w płaszczyźnie poziomej
            fill_mode='nearest'
            # strategia wypełniania nowo utworzonych pikseli, któe mogą powstać w wyniku przekształceń
        )

        return train_datagen

    def get_train_generator(self, train_dir):
        train_generator = self.datagen.flow_from_directory(directory=train_dir,
                                                           target_size=(224, 224),
                                                           batch_size=1,
                                                           class_mode='categorical')

        return train_generator

    def get_valid_generator(self, valid_dir):
        valid_datagen = ImageDataGenerator(rescale=1. / 255.)
        valid_generator = valid_datagen.flow_from_directory(directory=valid_dir,
                                                           target_size=(224, 224),
                                                           batch_size=1,
                                                           class_mode='categorical')

        return valid_generator

    def get_test_generator(self, test_dir):
        test_datagen = ImageDataGenerator(rescale=1. / 255.)
        test_generator = test_datagen.flow_from_directory(directory=test_dir,
                                                          target_size=(224, 224),
                                                          batch_size=1,
                                                          class_mode='categorical',
                                                          shuffle=False)

        return test_generator
