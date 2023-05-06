import os
import shutil
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report

from Backend.AI.AIModel import AIModel
from Backend.AI.Generators import Generator


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


def prepareDataForEachClass(classes: list, base_dir, train_dir, valid_dir, test_dir, train_size=0.6, valid_size=0.2,
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
                      yaxis_title='Accuracy', yaxis_type='linear')
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='loss', mode='markers+lines'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='val_loss', mode='markers+lines'))
    fig.update_layout(width=1000, height=500, title='Loss vs. Val Loss', xaxis_title='Epoki', yaxis_title='Loss',
                      yaxis_type='linear')
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

base_dir = '../../Banknotes'
data_dir = '../../images'

classes = getAllClasses(base_dir)
train_dir, valid_dir, test_dir = createBaseFoldersTree(data_dir)
train_size, valid_size, test_size = prepareDataForEachClass(classes=classes, base_dir=base_dir, train_dir=train_dir,
                                                            valid_dir=valid_dir, test_dir=test_dir)

generator = Generator()
train_generator = generator.get_train_generator(train_dir=train_dir)
valid_generator = generator.get_valid_generator(valid_dir=valid_dir)
test_generator = generator.get_test_generator(test_dir=test_dir)

batch_size = 1
steps_per_epoch = train_size // batch_size
validation_steps = valid_size // batch_size

ai_model = AIModel()

ai_model.load()
ai_model.model.summary()
# ai_model.buildModel(classesNo=len(classes), load_weights=False)
# ai_model.trainModel(train_generator=train_generator,
#                     valid_generator=valid_generator,
#                     train_size=train_size,
#                     valid_size=valid_size,
#                     batch_size=1,
#                     epochs=2000,
#                     save_heights=False)

# ai_model.save_model()
# ai_model.plot_hist()


# jeśli chcesz predykrtować plik przy pomocy ścieżki użyj tej funckji i podaj ścieżkę
path="../../Banknotes/Poland_100/34.jpg"
x = ai_model.predictByImagePath(path)
# Jeśli chcesz predyktowac plik przez podanie gotowego obrazu jako obiektu w python musi być przygotowany w następujący
# sposób:
#
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# img = Image.open(imagePath)
# img = np.array(img)
# img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
# img_tensor = tf.image.resize(img_tensor, [224, 224])
# img_tensor = tf.expand_dims(img_tensor, 0)
#

# Obie funkcje zwracją pozycję klasy którą rozpoznał model, więc odwołując się do naszykch klas możemy wyciągnąć nazwę
# tej klasy (patrz linijkę niżej):
print(classes[x])


filenames = test_generator.filenames
nb_samples = len(filenames)
y_prob = ai_model.model.predict(x=test_generator,
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

plot_confusion_matrix(cm)

print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

errors = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}, index=test_generator.filenames)
print(errors)

errors['is_incorrect'] = (errors['y_true'] != errors['y_pred']) * 1
print(errors)
