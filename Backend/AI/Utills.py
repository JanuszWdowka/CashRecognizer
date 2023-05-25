"""
Plik zawierający funkcje użytkowe dla modelu
"""
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage import io
from skimage.filters import sobel
from skimage.morphology import binary_closing, disk
from skimage.measure import label, regionprops, shannon_entropy
from scipy import ndimage
from scipy.stats import skew
import skimage
from skimage import io, transform
from skimage.io import imread, imsave
import os
import easyocr


def getBanknote(imagePath):
    """
    Funkcja znajdująca największy element na zdjęciu, który jest uznawany za banknot
    :param imagePath: ścieżka dostępu do zdjęcia
    :return: obrobione zdjęcie z wyciętym banknotem
    """
    # wczytaj obraz banknotu
    image = io.imread(imagePath)

    # przetwarzanie wstępne - zamiana na obraz w skali szarości i wygładzenie
    gray_image = skimage.color.rgb2gray(image[:, :, :3])
    blurred_image = skimage.filters.gaussian(gray_image, sigma=2.0)

    # wykrywanie krawędzi przy użyciu operatora Sobela
    edge_sobel = sobel(blurred_image)

    # binaryzacja i zamknięcie - przekształcenia morfologiczne służące do usunięcia małych artefaktów
    thresh = skimage.filters.threshold_li(edge_sobel)
    binary = edge_sobel > thresh
    closed = binary_closing(binary, disk(3))

    # segmentacja - wyodrębnienie pojedynczych obiektów
    label_image = label(closed)

    # wybierz największy obiekt (zakładając, że to banknot)
    regions = regionprops(label_image)
    areas = [r.area for r in regions]
    max_area_idx = np.argmax(areas)
    max_region = regions[max_area_idx]


    minr, minc, maxr, maxc = max_region.bbox

    # wycięcie banknotu
    banknote = image[minr:maxr, minc:maxc]


    # fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(10, 10))
    #
    # axes[0].imshow(image, cmap="gray")
    # axes[0].set_title("Obraz oryginalny")
    #
    # axes[1].imshow(banknote, cmap="gray")
    # axes[1].set_title("Obraz po wycięciu")
    # plt.show()

    return banknote


def prepareHistogram(image):
    """
    Funkcja wyliczająca histogram kanałów RGB dla zdjęcia
    :param image: ścieżka dostępu do zdjęcia
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # r_hist, r_bins = np.histogram(image[:, :, 0].ravel(), bins=256, range=[0, 256])
    # g_hist, g_bins = np.histogram(image[:, :, 1].ravel(), bins=256, range=[0, 256])
    # b_hist, b_bins = np.histogram(image[:, :, 2].ravel(), bins=256, range=[0, 256])

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #
    # ax[0].plot(r_hist, color='red', label='Red')
    # ax[0].plot(g_hist, color='green', label='Green')
    # ax[0].plot(b_hist, color='blue', label='Blue')
    #
    # ax[1].imshow(image, cmap="gray")

    # plt.show()


def getAvgRGB(image) -> (float, float, float):
    """
    Funkcja wyliczająca średnie natężenie kanałow RBG na zdjęci
    :param image: ścieżka dostępu do zdjęcia
    :return: średnie wartości kanałów
    """
    # Wczytanie obrazu
    img = image

    # Obliczenie średnich wartości natężenia RGB
    avg_r = img[:, :, 0].mean()
    avg_g = img[:, :, 1].mean()
    avg_b = img[:, :, 2].mean()

    colors = ['r', 'g', 'b']

    # Tworzenie wykresu
    # plt.bar(['Red', 'Green', 'Blue'], [avg_r, avg_g, avg_b], color=colors)
    # plt.xlabel('Color channel')
    # plt.ylabel('Average value')
    # plt.title('Average color values')
    # plt.show()

    # Wyświetlenie wyników
    # print(f"Średnie natężenie RGB: R={avg_r:.2f}, G={avg_g:.2f}, B={avg_b:.2f}")
    return int(avg_r), int(avg_g), int(avg_b)

def getProportion(image):
    """
    Funkcja wyliczająca proporcje długości do szerokości znalezionego banknotu
    :param image: ścieżka dostępu do zdjęcia
    :return: wartość propocji szerokości do długości
    """
    height, width, _ = image.shape

    return width / height

def getBanknoteValue(image):
    """
    Funkcja znajdująca wartość banknotu na zdjęciu
    :param image: ścieżka dostępu do zdjęcia
    :return: nominał banknotu
    """
    width, height = image.shape[1], image.shape[0]

    right_upper_corner = image[0:height//2, width//2:width]
    left_upper_corner = image[:height//3, :width//3]
    right_lower_corner = image[height//2:, width//2:]
    left_lower_corner = image[height//2:height, 0:width//3]

    numbers = getNumbersFromImage(image)
    numbers_right_upper = getNumbersFromImage(right_upper_corner)
    numbers_left_upper = getNumbersFromImage(left_upper_corner)
    numbers_right_lower = getNumbersFromImage(right_lower_corner)
    numbers_left_lower = getNumbersFromImage(left_lower_corner)
    value = 0

    if numbers and (numbers[0]%10 == 0 or numbers[0] == 2.0 or numbers[0] == 5.0 or numbers[0] == 1.0):
        value = numbers[0]
    elif numbers_right_upper and (numbers_right_upper[0]%10 == 0 or numbers_right_upper[0] == 2.0 or numbers_right_upper[0] == 5.0 or numbers_right_upper[0] == 1.0):
        value = numbers_right_upper[0]
    elif numbers_left_upper and (numbers_left_upper[0]%10 == 0 or numbers_left_upper[0] == 2.0 or numbers_left_upper[0] == 5.0 or numbers_left_upper[0] == 1.0):
        value = numbers_left_upper[0]
    elif numbers_right_lower and (numbers_right_lower[0]%10 == 0 or numbers_right_lower[0] == 2.0 or numbers_right_lower[0] == 5.0 or numbers_right_lower[0] == 1.0):
        value = numbers_right_lower[0]
    elif numbers_left_lower and (numbers_left_lower[0]%10 == 0 or numbers_left_lower[0] == 2.0 or numbers_left_lower[0] == 5.0 or numbers_left_lower[0] == 1.0):
        value = numbers_left_lower[0]
    
    return value
    

def getNumbersFromImage(image):
    """
    Funkcja znajdująca liczby na zdjęciu
    :param image: ścieżka dostępu do zdjęcia
    :return: znalezione liczby
    """
    reader = easyocr.Reader(['en'])

    result = reader.readtext(image)
    numbers = []
    
    for r in result:
        text = r[1]
        try:
            number = float(text)
            numbers.append(number)
        except ValueError:
            pass
    return numbers

def rotateImage(image):
    """
    Funkcja obracjąca zdjęcie o 90 stopni
    :param image: ścieżka dostępu do zdjęcia
    :return: obrócone zdjęcie
    """
    height, width, _ = image.shape

    if height > width:
        image = rotate(image, 90)
    
    return image

def getEntropy(image):
    """
    Funkcja obliczająca entropię na zdjęciu
    :param image: ścieżka dostępu do zdjęcia
    :return: wartość entropii
    """
    return shannon_entropy(image)

def getVariance(image):
    """
    Funkcja obliczająca wariację na zdjęciu
    :param image: ścieżka dostępu do zdjęcia
    :return: wartość wariacji
    """
    return ndimage.variance(image)

def getSkewness(image):
    """
    Funkcja obliczająca skośność na zdjęciu
    :param image: ścieżka dostępu do zdjęcia
    :return: wartość skośności
    """
    flatten_image = image.flatten()
    
    return skew(flatten_image)


def map_to_range(value):
    """
    Funkcja mapująca wartość do zakresu 0-255
    :param value: Wartość do zmapowania
    :return: Warość zmapowana
    """
    if value < 0:
        return 0
    elif value > 255:
        return 255
    else:
        return value


def map_range_1_to_0_255(x):
    """
    Funkcja mapująca wartość z zakresu -1 do 1 do zakresu 0 do 255
    :param x: Warość do zmapowania
    :return: Warość zmapowana
    """
    if x < -1:
        x = -1
    elif x > 1:
        x = 1
    przeskalowane = (x + 1) * 127.5
    zaokraglone = round(przeskalowane)
    return int(zaokraglone)

def resize_image(image):
    """
    Funkcja zmieniająca rozmiar zdjęcia na 224x224
    :param image_path:
    :return: Przeskalowane zdjęcie
    """
    resized_image = transform.resize(image, (224, 224), anti_aliasing=True)

    return resized_image

def saveOpenedFile(image, path, imageName):
    """
    Funkcja zapisująca zdjęcie w folderze
    :param image: Obraz do zapisania
    :param path: Ścieżka do folderu zapisu
    :param imageName: Nazwa nowego zdjęcia
    :return: Zwraca ścieżkę do nowego zdjęcia
    """

    newPath = os.path.join(path,imageName)

    imsave(newPath, image)

    return newPath

def prepareDataForImage(imagePath):
    """
    Funkcja przygotowująca dane z obrazu i zapisująca je na pixelach przeskalowanego obrazu
    :param imagePath: ścieżka do zdjęcia
    :return: przygotowane zdjęcie z naniesionymi danymi
    """
    banknote = getBanknote(imagePath)

    if banknote is None:
        return None

    banknote = rotateImage(banknote)

    avg_r, avg_g, avg_b = getAvgRGB(banknote)
    proportion = int(getProportion(banknote) * 100)
    if proportion > 255:
        proportion = 255
    # banknoteValue = int(getBanknoteValue(banknote))
    entropy = int(getEntropy(banknote))
    variance = int(sqrt(getVariance(banknote)))
    skewness = map_range_1_to_0_255(getSkewness(banknote))

    values = [avg_r, avg_g, avg_b, proportion, entropy, variance, skewness]


    # banknote = resize_image(banknote)

    if banknote.shape[2] == 4:
        banknote = banknote[:, :, :3]
    # Wybierz współrzędne pikseli, które chcesz nadpisać
    x_coords = [0, 0, 0, 0, 0, 0, 0]
    y_coords = [0, 10, 20, 30, 40, 50, 60]

    for y, value in zip(y_coords,values):
        # value = int(value) / 255.0
        # Przypisz nową wartość pikselom
        new_value = [value, value, value]
        for x in x_coords:
            banknote[y:y+10, x:x+10] = new_value


    return banknote

def createNewImageWithData(imagePath):
    """
    Funkcja tworząca nowe zdjęcie z naniesionymi danymi
    :param imagePath: ścieżka do zdjęcia
    :return: ścieżka do nowego zdjęcia
    """
    banknote = prepareDataForImage(imagePath)

    if banknote is None:
        return None

    fileName = os.path.basename(imagePath)
    path = os.path.dirname(imagePath)
    path = path.replace('Banknotes', 'BanknotesV2')
    if not os.path.exists(path):
        os.makedirs(path)

    newPath = saveOpenedFile(banknote, path, fileName)

    return newPath

def found_all_files( folder):
    """
    Funkcja zwraca listę ścieżek do wszystkich plików w folderze
    :param folder: Ścieżka do folderu
    :return: Lista ścieżek do plików
    """
    lista_sciezek = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".jpg"):
                sciezka = os.path.join(root, file)
                lista_sciezek.append(sciezka)
    return lista_sciezek

def prepareImagesWithFeatures( pathToFolderWithImages):
    """
    Funkcja przygotowująca zdjęcia z danymi wraz z cecami banknotów z podanego folderu
    :param pathToFolderWithImages:
    :return:
    """
    x = 0
    files_paths = found_all_files(pathToFolderWithImages)
    for file_path in files_paths:
        print(file_path)
        createNewImageWithData(file_path)
        x = x + 1
        print(rf"Wykonano: {x} / {len(files_paths)}")


# prepareImagesWithFeatures('/Users/adamludwiczak/PycharmProjects/AnalizaDanych/CashRecognizer/Banknotes')