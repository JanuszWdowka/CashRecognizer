import os


def createFolder(path):
    """
    Function creates folder in given path
    :param path:
    :return: Path created folder
    """
    import os
    if not os.path.exists(path):
        os.mkdir(path)

    return path


def deleteFolder(path):
    """
    Function deletes folder in given path
    :param path:
    :return: None
    """
    import os
    if os.path.exists(path):
        os.remove(path)
    os.remove(path)

def createBaseFoldersTree(path):
    """
    Function creates base folders tree
    :param path:
    :return: Folders for train, valid and test sets
    """
    path = createFolder(path)
    train_dir = createFolder(os.path.join(path, 'train'))  # katalog zbioru treningowego
    valid_dir = createFolder(os.path.join(path, 'valid'))  # katalog zbioru walidacyjnego
    test_dir = createFolder(os.path.join(path, 'test'))  # katalog zbioru testowego

    return train_dir, valid_dir, test_dir

def getAllClasses(path):
    """
    Function returns all folders in given path
    :param path:
    :return: List of folders names
    """
    folder_names = []
    for entry in os.scandir(path):
        if entry.is_dir():
            folder_names.append(entry.name)
    return folder_names