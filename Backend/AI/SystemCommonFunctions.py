"""
Logika używana do przygotowania danych.
"""
import os

def createFolder(path):
    """
    Funkcja tworzy folder w podanej ścieżce
    :param path: Ścieżka do folderu
    :return: Ścieżka do folderu
    """
    import os
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def deleteFolder(path):
    """
    Funkcja usuwa folder w podanej ścieżce
    :param path: Ścieżka do folderu
    """
    import os
    if os.path.exists(path):
        os.remove(path)
    os.remove(path)

def createBaseFoldersTree(path):
    """
    Funkcja tworzy foldery do przechowywania danych do trenowania, walidacji i testowania
    :param path: Ścieżka do folderu
    :return: Ścieżki do folderów z danymi do trenowania, walidacji i testowania
    """
    path = createFolder(path)
    train_dir = createFolder(os.path.join(path, 'train'))  # katalog zbioru treningowego
    valid_dir = createFolder(os.path.join(path, 'valid'))  # katalog zbioru walidacyjnego
    test_dir = createFolder(os.path.join(path, 'test'))  # katalog zbioru testowego

    return train_dir, valid_dir, test_dir

def getAllClasses(path):
    """
    Funkcja zwraca listę nazw klas
    :param path: Ścieżka do folderu skąd będą pobierane klasy
    :return: Lista nazw klas
    """
    folder_names = []
    for entry in os.scandir(path):
        if entry.is_dir():
            folder_names.append(entry.name)
    return folder_names