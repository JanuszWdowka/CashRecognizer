import os
import shutil

import numpy as np

from Backend.AI.SystemCommonFunctions import getAllClasses, createFolder, createBaseFoldersTree


class DataPreparer:
    """
    Klasa przygotowująca dane do trenowania, walidacji i testowania
    """
    def __init__(self, base_dir, data_dir):
        """
        Konstrukrtor klasy DataPreparer
        :param base_dir: Ścieżka do plików z danymi
        :param data_dir: Ścieżkd do folderu gdzie zostaną umieszczone dane do trenowania, walidacji i testowania
        """
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.train_dir, self.valid_dir, self.test_dir = createBaseFoldersTree(data_dir)
        self.classes = getAllClasses(path=base_dir)

    def __countMinItems(self, base_dir):
        """
        Funkcja zwraca minimalną liczbę obrazów ze zbioru klas
        :param base_dir: Ścieżka do plików z danymi
        :return: Minimalna liczba obrazów ze zbioru klas
        """
        items = list()
        for itemClass in self.classes:
            x = os.listdir(os.path.join(base_dir, itemClass))
            items.append(len([fname for fname in x if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]))

        return min(items)

    def getClasses(self):
        """
        Funkcja pobierająca wszystkie klasy
        :return: Zwraza listę znalezionych klas
        """
        return self.classes

    def prepareDataForEachClass(self, train_size=0.6, valid_size=0.2, info=False):
        """
        Funckja przygotowuje dane dla każdej klasy
        :param base_dir: Ścieżka do plików z danymi
        :param train_dir: Scieżka do folderu z danymi do trenowania
        :param valid_dir: Ścieżka do folderu z danymi do walidacji
        :param test_dir: Ścieżka do folderu z danymi do testowania
        :param train_size: Procent danych do trenowania
        :param valid_size: Procent danych do walidacji
        :param info: Informacja debugująca
        :return: Zwraza liczbę danych do trenowania, walidacji i testowania
        """
        base_dir = self.base_dir
        train_dir = self.train_dir
        valid_dir = self.valid_dir
        test_dir = self.test_dir
        size = self.__countMinItems(base_dir=base_dir)

        train_size_counted = int(np.floor(train_size * size))
        valid_size_counted = int(np.floor(valid_size * size))
        test_size_counted = size - train_size_counted - valid_size_counted

        train_idx = train_size_counted
        valid_idx = train_size_counted + valid_size_counted
        test_idx = train_size_counted + valid_size_counted + test_size_counted

        for itemClass in self.classes:
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


    def getDirs(self):
        """
        Funkcja zwraca ścieżki do folderów z danymi do trenowania, walidacji i testowania
        :return: Zwraca ścieżki do folderów z danymi do trenowania, walidacji i testowania
        """
        return self.train_dir, self.valid_dir, self.test_dir


