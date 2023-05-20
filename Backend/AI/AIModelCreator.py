import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from Backend.AI.AIModel import AIModel
from Backend.AI.CommonFunctions import plot_confusion_matrix
from Backend.AI.DataPreparer import DataPreparer
from Backend.AI.Generators import Generator

"""
Skrypt do trenowania, tworzenia, testowania i walidacji modelu
"""

option = "create"
base_dir = '../../Banknotes'
data_dir = '../../images'
np.set_printoptions(precision=6, suppress=True)
ai_model = AIModel()
generator = Generator()

dataPreparer = DataPreparer(base_dir=base_dir, data_dir=data_dir)

train_size, valid_size, test_size = dataPreparer.prepareDataForEachClass()
train_dir, valid_dir, test_dir = dataPreparer.getDirs()


train_generator = generator.get_train_generator(train_dir=train_dir)
valid_generator = generator.get_valid_generator(valid_dir=valid_dir)
test_generator = generator.get_test_generator(test_dir=test_dir)

batch_size = 1
steps_per_epoch = train_size // batch_size
validation_steps = valid_size // batch_size

if option == "create":
    ai_model.buildModel(classesNo=len(dataPreparer.getClasses()))
    ai_model.modelSummary()
    ai_model.trainModel(train_generator=train_generator,
                        valid_generator=valid_generator,
                        train_size=train_size,
                        valid_size=valid_size,
                        batch_size=1,
                        epochs=10000,
                        save_heights=False)

    ai_model.save_model()
    ai_model.plot_hist()
elif option == "load":
    ai_model.load(modelPath="./model_data.h5")
    ai_model.model.summary()
else:
    print("Wrong option")

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

plot_confusion_matrix(cm, classes)

print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

errors = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}, index=test_generator.filenames)
print(errors)

errors['is_incorrect'] = (errors['y_true'] != errors['y_pred']) * 1
print(errors)
