import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import plotly.graph_objects as go
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.callbacks import TensorBoard
from keras.applications import VGG19, VGG16

def plot_hist(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='accuracy', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='val_accuracy', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='Accuracy vs. Val Accuracy', xaxis_title='Epoki', yaxis_title='Accuracy', yaxis_type='log')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='loss', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='val_loss', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='Loss vs. Val Loss', xaxis_title='Epoki', yaxis_title='Loss', yaxis_type='log')
    fig.show()




np.set_printoptions(precision=6, suppress=True)

base_dir = '/Users/adamludwiczak/PycharmProjects/AnalizaDanych/CashRecognizer/Banknotes'
raw_no_of_files = {}
classes = ['Poland_10', 'Poland_20', 'Poland_50', 'Poland_100', 'Poland_200', 'Poland_500']
for dir in classes:
    raw_no_of_files[dir] = len(os.listdir(os.path.join(base_dir, dir)))

raw_no_of_files.items()

data_dir = '../images'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

train_dir = os.path.join(data_dir, 'train')    # katalog zbioru treningowego
valid_dir = os.path.join(data_dir, 'valid')    # katalog zbioru walidacyjnego
test_dir = os.path.join(data_dir, 'test')      # katalog zbioru testowego

train_Poland_10_dir     =   os.path.join(train_dir,     'Poland_10')
train_Poland_20_dir     =   os.path.join(train_dir,     'Poland_20')
train_Poland_50_dir     =   os.path.join(train_dir,     'Poland_50')
train_Poland_100_dir    =   os.path.join(train_dir,     'Poland_100')
train_Poland_200_dir    =   os.path.join(train_dir,     'Poland_200')
train_Poland_500_dir    =   os.path.join(train_dir,     'Poland_500')

valid_Poland_10_dir     =   os.path.join(valid_dir,     'Poland_10')
valid_Poland_20_dir     =   os.path.join(valid_dir,     'Poland_20')
valid_Poland_50_dir     =   os.path.join(valid_dir,     'Poland_50')
valid_Poland_100_dir    =   os.path.join(valid_dir,     'Poland_100')
valid_Poland_200_dir    =   os.path.join(valid_dir,     'Poland_200')
valid_Poland_500_dir    =   os.path.join(valid_dir,     'Poland_500')

test_Poland_10_dir      =   os.path.join(test_dir,      'Poland_10')
test_Poland_20_dir      =   os.path.join(test_dir,      'Poland_20')
test_Poland_50_dir      =   os.path.join(test_dir,      'Poland_50')
test_Poland_100_dir     =   os.path.join(test_dir,      'Poland_100')
test_Poland_200_dir     =   os.path.join(test_dir,      'Poland_200')
test_Poland_500_dir     =   os.path.join(test_dir,      'Poland_500')

for directory in (train_dir, valid_dir, test_dir):
    if not os.path.exists(directory):
        os.mkdir(directory)

dirs = [
train_Poland_10_dir ,
train_Poland_20_dir ,
train_Poland_50_dir ,
train_Poland_100_dir,
train_Poland_200_dir,
train_Poland_500_dir,
valid_Poland_10_dir ,
valid_Poland_20_dir ,
valid_Poland_50_dir ,
valid_Poland_100_dir,
valid_Poland_200_dir,
valid_Poland_500_dir,
test_Poland_10_dir  ,
test_Poland_20_dir  ,
test_Poland_50_dir  ,
test_Poland_100_dir ,
test_Poland_200_dir ,
test_Poland_500_dir ,
]

for dir in dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)

Poland_10_fnames    = os.listdir(os.path.join(base_dir, 'Poland_10'))
Poland_20_fnames    = os.listdir(os.path.join(base_dir, 'Poland_20'))
Poland_50_fnames    = os.listdir(os.path.join(base_dir, 'Poland_50'))
Poland_100_fnames   = os.listdir(os.path.join(base_dir, 'Poland_100'))
Poland_200_fnames   = os.listdir(os.path.join(base_dir, 'Poland_200'))
Poland_500_fnames   = os.listdir(os.path.join(base_dir, 'Poland_500'))

Poland_10_fnames = [fname for fname in Poland_10_fnames if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
Poland_20_fnames = [fname for fname in Poland_20_fnames if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
Poland_50_fnames = [fname for fname in Poland_50_fnames if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
Poland_100_fnames = [fname for fname in Poland_100_fnames if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
Poland_200_fnames = [fname for fname in Poland_200_fnames if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]
Poland_500_fnames = [fname for fname in Poland_500_fnames if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]

size = min(len(Poland_10_fnames),
           len(Poland_20_fnames),
           len(Poland_50_fnames),
           len(Poland_100_fnames),
           len(Poland_200_fnames),
           len(Poland_500_fnames))

train_size = int(np.floor(0.5 * size))
valid_size = int(np.floor(0.2 * size))
test_size = size - train_size - valid_size

train_idx = train_size
valid_idx = train_size + valid_size
test_idx = train_size + valid_size + test_size

for i, fname in enumerate(Poland_10_fnames):
    if i <= train_idx:
        src = os.path.join(base_dir, 'Poland_10', fname)
        dst = os.path.join(train_Poland_10_dir, fname)
        shutil.copyfile(src, dst)
    elif train_idx < i <= valid_idx:
        src = os.path.join(base_dir, 'Poland_10', fname)
        dst = os.path.join(valid_Poland_10_dir, fname)
        shutil.copyfile(src, dst)
    elif valid_idx < i < test_idx:
        src = os.path.join(base_dir, 'Poland_10', fname)
        dst = os.path.join(test_Poland_10_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(Poland_20_fnames):
    if i <= train_idx:
        src = os.path.join(base_dir, 'Poland_20', fname)
        dst = os.path.join(train_Poland_20_dir, fname)
        shutil.copyfile(src, dst)
    elif train_idx < i <= valid_idx:
        src = os.path.join(base_dir, 'Poland_20', fname)
        dst = os.path.join(valid_Poland_20_dir, fname)
        shutil.copyfile(src, dst)
    elif valid_idx < i < test_idx:
        src = os.path.join(base_dir, 'Poland_20', fname)
        dst = os.path.join(test_Poland_20_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(Poland_50_fnames):
    if i <= train_idx:
        src = os.path.join(base_dir, 'Poland_50', fname)
        dst = os.path.join(train_Poland_50_dir, fname)
        shutil.copyfile(src, dst)
    elif train_idx < i <= valid_idx:
        src = os.path.join(base_dir, 'Poland_50', fname)
        dst = os.path.join(valid_Poland_50_dir, fname)
        shutil.copyfile(src, dst)
    elif valid_idx < i < test_idx:
        src = os.path.join(base_dir, 'Poland_50', fname)
        dst = os.path.join(test_Poland_50_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(Poland_100_fnames):
    if i <= train_idx:
        src = os.path.join(base_dir, 'Poland_100', fname)
        dst = os.path.join(train_Poland_100_dir, fname)
        shutil.copyfile(src, dst)
    elif train_idx < i <= valid_idx:
        src = os.path.join(base_dir, 'Poland_100', fname)
        dst = os.path.join(valid_Poland_100_dir, fname)
        shutil.copyfile(src, dst)
    elif valid_idx < i < test_idx:
        src = os.path.join(base_dir, 'Poland_100', fname)
        dst = os.path.join(test_Poland_100_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(Poland_200_fnames):
    if i <= train_idx:
        src = os.path.join(base_dir, 'Poland_200', fname)
        dst = os.path.join(train_Poland_200_dir, fname)
        shutil.copyfile(src, dst)
    elif train_idx < i <= valid_idx:
        src = os.path.join(base_dir, 'Poland_200', fname)
        dst = os.path.join(valid_Poland_200_dir, fname)
        shutil.copyfile(src, dst)
    elif valid_idx < i < test_idx:
        src = os.path.join(base_dir, 'Poland_200', fname)
        dst = os.path.join(test_Poland_200_dir, fname)
        shutil.copyfile(src, dst)

for i, fname in enumerate(Poland_500_fnames):
    if i <= train_idx:
        src = os.path.join(base_dir, 'Poland_500', fname)
        dst = os.path.join(train_Poland_500_dir, fname)
        shutil.copyfile(src, dst)
    elif train_idx < i <= valid_idx:
        src = os.path.join(base_dir, 'Poland_500', fname)
        dst = os.path.join(valid_Poland_500_dir, fname)
        shutil.copyfile(src, dst)
    elif valid_idx < i < test_idx:
        src = os.path.join(base_dir, 'Poland_500', fname)
        dst = os.path.join(test_Poland_500_dir, fname)
        shutil.copyfile(src, dst)

print('Poland_10 - zbiór treningowy',       len(os.listdir(train_Poland_10_dir  )))
print('Poland_10 - zbiór walidacyjny',      len(os.listdir(valid_Poland_10_dir  )))
print('Poland_10 - zbiór testowy',          len(os.listdir(test_Poland_10_dir   )))

print('Poland_20 - zbiór treningowy',   len(os.listdir(train_Poland_20_dir)))
print('Poland_20 - zbiór walidacyjny',  len(os.listdir(valid_Poland_20_dir)))
print('Poland_20 - zbiór testowy',      len(os.listdir(test_Poland_20_dir)))

print('Poland_50 - zbiór treningowy',  len(os.listdir(train_Poland_50_dir)))
print('Poland_50 - zbiór walidacyjny', len(os.listdir(valid_Poland_50_dir)))
print('Poland_50 - zbiór testowy',     len(os.listdir(test_Poland_50_dir)))

print('Poland_100 - zbiór treningowy',  len(os.listdir(train_Poland_100_dir)))
print('Poland_100 - zbiór walidacyjny', len(os.listdir(valid_Poland_100_dir)))
print('Poland_100 - zbiór testowy',     len(os.listdir(test_Poland_100_dir)))

print('Poland_200 - zbiór treningowy',  len(os.listdir(train_Poland_200_dir)))
print('Poland_200 - zbiór walidacyjny', len(os.listdir(valid_Poland_200_dir)))
print('Poland_200 - zbiór testowy',     len(os.listdir(test_Poland_200_dir)))

print('Poland_500 - zbiór treningowy',  len(os.listdir(train_Poland_500_dir)))
print('Poland_500 - zbiór walidacyjny', len(os.listdir(valid_Poland_500_dir)))
print('Poland_500 - zbiór testowy',     len(os.listdir(test_Poland_500_dir)))

train_datagen = ImageDataGenerator(
    rotation_range=10,      # zakres kąta o który losowo zostanie wykonany obrót obrazów
    rescale=1./255.,
    width_shift_range=0.2,  # pionowe przekształcenia obrazu
    height_shift_range=0.2, # poziome przekształcenia obrazu
    shear_range=0.2,        # zares losowego przycianania obrazu
    zoom_range=0.2,         # zakres losowego przybliżania obrazu
    horizontal_flip=True,   # losowe odbicie połowy obrazu w płaszczyźnie poziomej
    fill_mode='nearest'     # strategia wypełniania nowo utworzonych pikseli, któe mogą powstać w wyniku przekształceń
)

# przeskalowujemy wszystkie obrazy o współczynnik 1/255
valid_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(directory=valid_dir,
                                                   target_size=(224, 224),
                                                   batch_size=32,
                                                   class_mode='categorical')

batch_size = 32
steps_per_epoch = train_size // batch_size
validation_steps = valid_size // batch_size

conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_base.trainable = True

def print_layers(model):
    for layer in model.layers:
        print(f'layer_name: {layer.name:13} trainable: {layer.trainable}')

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

print_layers(conv_base)

model = Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(units=256, activation='relu'))
model.add(layers.Dense(units=6, activation='softmax'))

# optimizer = RMSprop(learning_rate=1e-5)

model.compile(optimizer=RMSprop(lr=1e-5),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.build((None, 224, 224, 3))
model.summary()

history = model.fit(x=train_generator,
                    epochs=10,
                    steps_per_epoch=10,
                    validation_data=valid_generator,
                    validation_steps=validation_steps)

# plot_hist(history)

test_datagen = ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

y_prob = model.predict_generator(test_generator, test_generator.samples)
y_prob



