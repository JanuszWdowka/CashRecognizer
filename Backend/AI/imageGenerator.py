from keras_preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from numpy import expand_dims

img = load_img('/Users/adamludwiczak/PycharmProjects/AnalizaDanych/CashRecognizer/Banknotes/Euro_5/1-8.jpg')
data = img_to_array(img)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(width_shift_range=0.2,
                             height_shift_range=0.2,
                             rotation_range=10,
                             brightness_range=[0.4, 1.5],
                             shear_range=20,
                             zoom_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=True)

iterator = datagen.flow(samples, batch_size=1)

plt.imshow(img)
plt.show()

plt.figure(figsize=(16, 8))
for i in range(9):
    plt.subplot(330 + i + 1)
    batch = iterator.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()
