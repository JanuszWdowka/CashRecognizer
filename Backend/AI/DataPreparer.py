import os
import shutil

import numpy as np

from Backend.AI.SystemCommonFunctions import getAllClasses, createFolder, createBaseFoldersTree


class DataPreparer:
    """
    Class prepares data for AI model
    """
    def __init__(self, base_dir, data_dir):
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.train_dir, self.valid_dir, self.test_dir = createBaseFoldersTree(data_dir)
        self.classes = getAllClasses(path=base_dir)

    def __countMinItems(self, base_dir):
        """
        Function counts minimum number of items in each class
        :param base_dir:
        :return:
        """
        items = list()
        for itemClass in self.classes:
            x = os.listdir(os.path.join(base_dir, itemClass))
            items.append(len([fname for fname in x if fname.split('.')[1].lower() in ['jpg', 'png', 'jpeg']]))

        return min(items)

    def getClasses(self):
        """
        Function returns list of classes
        :return:
        """
        return self.classes

    def prepareDataForEachClass(self, train_size=0.6, valid_size=0.2, info=False):
        """
        Function prepares data for each class in base_dir and saves it in train_dir, valid_dir and test_dir
        :param base_dir:
        :param train_dir:
        :param valid_dir:
        :param test_dir:
        :param train_size:
        :param valid_size:
        :param info:
        :return:
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


    def getDirs(self):
        return self.train_dir, self.valid_dir, self.test_dir


