from keras.preprocessing.image import ImageDataGenerator


class Generator:
    """
    Klasa generująca różne warianty zdjęcia do nauki dla modelu.
    """
    def __init__(self):
        self.datagen = self.__create_datagen()

    def __create_datagen(self):
        """
        Funkcja generująca warianty zdjęć dla modelu sztucznej inteligencji.
        :return: wygenerowane zdjęcia do nauki
        """
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
        """
        Funkcja służąca do pobranie zdjęć do trenowania modelu
        :param train_dir: ścieżka do folderu z plikami banknotów do trenowania modelu
        :return: pliki do trenowania
        """
        train_generator = self.datagen.flow_from_directory(directory=train_dir,
                                                           target_size=(224, 224),
                                                           batch_size=1,
                                                           class_mode='categorical')

        return train_generator

    def get_valid_generator(self, valid_dir):
        """
        Funkcja służąca do pobranie zdjęć do walidacji modelu
        :param valid_dir: ścieżka do folderu z plikami banknotów do walidacji modelu
        :return: pliki do walidacji
        """
        valid_datagen = ImageDataGenerator(rescale=1. / 255.)
        valid_generator = valid_datagen.flow_from_directory(directory=valid_dir,
                                                           target_size=(224, 224),
                                                           batch_size=1,
                                                           class_mode='categorical')

        return valid_generator

    def get_test_generator(self, test_dir):
        """
        Funkcja służąca do pobranie zdjęć do testu modelu
        :param test_dir: ścieżka do folderu z plikami banknotów do testu modelu
        :return: pliki do testu
        """
        test_datagen = ImageDataGenerator(rescale=1. / 255.)
        test_generator = test_datagen.flow_from_directory(directory=test_dir,
                                                          target_size=(224, 224),
                                                          batch_size=1,
                                                          class_mode='categorical',
                                                          shuffle=False)

        return test_generator
