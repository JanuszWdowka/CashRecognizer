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


def createFolder(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def deleteFolder(path):
    import os
    if os.path.exists(path):
        os.remove(path)
    os.remove(path)


def getAllClasses(path):
    folder_names = []
    for entry in os.scandir(path):
        if entry.is_dir():
            folder_names.append(entry.name)
    return folder_names


def createBaseFoldersTree(path):
    path = createFolder(path)
    train_dir = createFolder(os.path.join(path, 'train'))  # katalog zbioru treningowego
    valid_dir = createFolder(os.path.join(path, 'valid'))  # katalog zbioru walidacyjnego
    test_dir = createFolder(os.path.join(path, 'test'))  # katalog zbioru testowego

    return train_dir, valid_dir, test_dir


def countMinItems(classes, base_dir):
    items = list()
    for itemClass in classes:
        x = os.listdir(os.path.join(base_dir, itemClass))
        items.append(len([fname for fname in x if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]))

    return min(items)


def prepareDataForEachClass(classes: list, base_dir, train_dir, valid_dir, test_dir, train_size=0.5, valid_size=0.2,
                            info=False):
    size = countMinItems(classes=classes, base_dir=base_dir)
    train_size_counted = int(np.floor(train_size * size))
    valid_size_counted = int(np.floor(valid_size * size))
    test_size_counted = size - train_size_counted - valid_size_counted

    train_idx = train_size_counted
    valid_idx = train_size_counted + valid_size_counted
    test_idx = train_size_counted + valid_size_counted + test_size_counted

    for itemClass in classes:
        # Prepare folders for each class
        train_dir_for_class = createFolder(os.path.join(train_dir, itemClass))
        valid_dir_for_class = createFolder(os.path.join(valid_dir, itemClass))
        test_dir_for_class = createFolder(os.path.join(test_dir, itemClass))
        fnames = os.listdir(os.path.join(base_dir, itemClass))
        fnames = [fname for fname in fnames if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]

        for i, fname in enumerate(fnames):
            if i <= train_idx:
                src = os.path.join(base_dir, itemClass, fname)
                dst = os.path.join(train_dir_for_class, fname)
                shutil.copyfile(src, dst)
            elif train_idx < i <= valid_idx:
                src = os.path.join(base_dir, itemClass, fname)
                dst = os.path.join(valid_dir_for_class, fname)
                shutil.copyfile(src, dst)
            elif valid_idx < i < test_idx:
                src = os.path.join(base_dir, itemClass, fname)
                dst = os.path.join(test_dir_for_class, fname)
                shutil.copyfile(src, dst)

        if info:
            print(f'{itemClass} - zbiór treningowy', len(os.listdir(train_dir_for_class)))
            print(f'{itemClass} - zbiór walidacyjny', len(os.listdir(valid_dir_for_class)))
            print(f'{itemClass} - zbiór testowy', len(os.listdir(test_dir_for_class)))

    return train_size_counted, valid_size_counted, test_size_counted


def plot_hist(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='accuracy', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='val_accuracy', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='Accuracy vs. Val Accuracy', xaxis_title='Epoki',
                      yaxis_title='Accuracy', yaxis_type='log')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='loss', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='val_loss', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='Loss vs. Val Loss', xaxis_title='Epoki', yaxis_title='Loss',
                      yaxis_type='log')
    fig.show()


def print_layers(model):
    for layer in model.layers:
        print(f'layer_name: {layer.name:13} trainable: {layer.trainable}')


def plot_confusion_matrix(cm):
    # Mulitclass classification, 3 classes
    cm = cm[::-1]
    cm = pd.DataFrame(cm, columns=classes, index=classes[::-1])

    fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index), colorscale='ice',
                                      showscale=True, reversescale=True)
    fig.update_layout(width=500, height=500, title='Confusion Matrix', font_size=16)
    fig.show()


np.set_printoptions(precision=6, suppress=True)

base_dir = '../Banknotes'
data_dir = '../images'

classes = getAllClasses(base_dir)
train_dir, valid_dir, test_dir = createBaseFoldersTree(data_dir)
train_size, valid_size, test_size = prepareDataForEachClass(classes=classes, base_dir=base_dir, train_dir=train_dir,
                                                            valid_dir=valid_dir, test_dir=test_dir)


train_datagen = ImageDataGenerator(
    rotation_range=10,  # zakres kąta o który losowo zostanie wykonany obrót obrazów
    rescale=1. / 255.,
    width_shift_range=0.2,  # pionowe przekształcenia obrazu
    height_shift_range=0.2,  # poziome przekształcenia obrazu
    shear_range=0.2,  # zares losowego przycianania obrazu
    zoom_range=0.2,  # zakres losowego przybliżania obrazu
    horizontal_flip=True,  # losowe odbicie połowy obrazu w płaszczyźnie poziomej
    fill_mode='nearest'  # strategia wypełniania nowo utworzonych pikseli, któe mogą powstać w wyniku przekształceń
)

# przeskalowujemy wszystkie obrazy o współczynnik 1/255
valid_datagen = ImageDataGenerator(rescale=1. / 255.)

train_generator = train_datagen.flow_from_directory(directory=train_dir,
                                                    target_size=(600, 300),
                                                    batch_size=1,
                                                    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(directory=valid_dir,
                                                    target_size=(600, 300),
                                                    batch_size=1,
                                                    class_mode='categorical')

batch_size = 1
steps_per_epoch = train_size // batch_size
validation_steps = valid_size // batch_size

conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(600, 300, 3))
conv_base.trainable = True

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

optimizer = optimizers.rmsprop_v2.RMSprop(learning_rate=1e-5)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.build((None, 600, 300, 3))
model.summary()

history = model.fit(x=train_generator,
                    epochs=30,
                    steps_per_epoch=200,
                    validation_data=valid_generator,
                    validation_steps=validation_steps)

plot_hist(history)

test_datagen = ImageDataGenerator(rescale=1. / 255.)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(600, 300),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

filenames = test_generator.filenames
nb_samples = len(filenames)
y_prob = model.predict(x=test_generator,
                       steps=nb_samples)
print(y_prob)

y_pred = np.argmax(y_prob, axis=1)
print(y_pred)

predictions = pd.DataFrame({'class': y_pred})
print(predictions)

y_true = test_generator.classes
print(y_true)

y_pred = predictions['class'].values
print(y_pred)

print(test_generator.class_indices)

classes = list(test_generator.class_indices.keys())
print(classes)

cm = confusion_matrix(y_true, y_pred)
cm

plot_confusion_matrix(cm)

print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

errors = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}, index=test_generator.filenames)
print(errors)

errors['is_incorrect'] = (errors['y_true'] != errors['y_pred']) * 1
print(errors)
