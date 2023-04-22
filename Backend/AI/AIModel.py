from os.path import exists
import pandas as pd
import plotly.graph_objects as go
from keras.applications import VGG19
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import Sequential, load_model


class AIModel:
    def __init__(self):
        self.weights_path = 'weights-{epoch:02d}-val_accuracy_{val_accuracy:.4f}-val_loss_{val_loss:.4f}.ckpt'
        self.model_path = "model_data.h5"
        self.model = None
        self.history = None


    def save_model(self):
        try:
            if self.model is None:
                raise Exception("Model nie został zdefiniowany")
            print(self.model.summary())
            self.model.save(self.model_path)

        except Exception as e:
            print("blad zapisu")
            print(str(e))

    def load(self):
        try:
            print(self.model_path)
            print(self.model)
            self.model = load_model('./model_data.h5')
            print("2")

        except Exception as e:
            print("Bład podczas ladowania modelu")
            print(str(e))

    def plot_hist(self):
        try:
            if self.history is None:
                raise Exception("History nie istnieje")

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
            fig.update_layout(width=1000, height=500, title='Loss vs. Val Loss', xaxis_title='Epoki',
                              yaxis_title='Loss',
                              yaxis_type='linear')
            fig.show()
        except Exception:
            print("Coś poszło nie tak")

        finally:
            pass

    def modelSummary(self):
        try:
            if self.model is None:
                raise Exception("Model nie został zdefiniowany")

            self.model.summary()

        except Exception:
            print("Coś poszło nie tak!")

    def buildModel(self, classesNo, load_weights = False):
        try:
            conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            conv_base.trainable = True

            set_trainable = False
            for layer in conv_base.layers:
                if layer.name == 'block4_conv1':
                    set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False
            model = Sequential()

            model.add(conv_base)
            model.add(layers.Flatten())
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(units=256, activation='relu'))
            model.add(layers.Dense(units=classesNo, activation='softmax'))
            optimizer = optimizers.rmsprop_v2.RMSprop(learning_rate=1e-5)

            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            if exists('weights-118-val_accuracy_0.9524-val_loss_0.1476.ckpt.index') and load_weights:
                model.load_weights('weights-118-val_accuracy_0.9524-val_loss_0.1476.ckpt.index')

            model.build((None, 224, 224, 3))

            self.model = model

        except Exception:
            print("Budowa modelu nie powiodła się")

    def trainModel(self, train_generator, valid_generator, train_size, valid_size, batch_size=1, epochs=20, save_heights = False):
        try:
            if self.model is None:
                raise Exception("Model nie został zdefiniowany")

            steps_per_epoch = train_size // batch_size
            validation_steps = valid_size // batch_size

            earlyStop = EarlyStopping(monitor='val_loss',
                                      patience=3,
                                      restore_best_weights=True)
            checkpoint = ModelCheckpoint(filepath=self.weights_path,
                                         save_weights_only=True,
                                         monitor='val_loss',
                                         mode='min',
                                         save_best_only=True)
            callbacks = None
            if save_heights:
                callbacks = [checkpoint]

            self.history = self.model.fit(x=train_generator,
                                          epochs=epochs,
                                          steps_per_epoch=steps_per_epoch,
                                          validation_data=valid_generator,
                                          validation_steps=validation_steps,
                                          callbacks=callbacks)

        except Exception:
            print("Coś poszło nie tak!")

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'num_layers': self.num_layers,
            'units': self.units,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
        })
        return config