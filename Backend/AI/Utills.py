import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import sobel
from skimage.morphology import binary_closing, disk
from skimage.measure import label, regionprops, shannon_entropy
from scipy import ndimage
from scipy.stats import skew
import skimage
import easyocr


def getBanknote(imagePath):
    # wczytaj obraz banknotu
    image = io.imread(imagePath)

    # przetwarzanie wstępne - zamiana na obraz w skali szarości i wygładzenie
    gray_image = skimage.color.rgb2gray(image)
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


    fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(10, 10))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Obraz oryginalny")

    axes[1].imshow(banknote, cmap="gray")
    axes[1].set_title("Obraz po wycięciu")
    plt.show()

    return banknote


def map_to_255(y):
    y_min, y_max = np.min(y), np.max(y)
    return (y - y_min) * 255 / (y_max - y_min)

def prepareHistogram(image):
    import numpy as np
    import matplotlib.pyplot as plt

    r_hist, r_bins = np.histogram(image[:, :, 0].ravel(), bins=256, range=[0, 256])
    g_hist, g_bins = np.histogram(image[:, :, 1].ravel(), bins=256, range=[0, 256])
    b_hist, b_bins = np.histogram(image[:, :, 2].ravel(), bins=256, range=[0, 256])

    # r_hist = map_to_255(r_hist)
    # g_hist = map_to_255(g_hist)
    # b_hist = map_to_255(b_hist)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(r_hist, color='red', label='Red')
    ax[0].plot(g_hist, color='green', label='Green')
    ax[0].plot(b_hist, color='blue', label='Blue')

    ax[1].imshow(image, cmap="gray")

    plt.show()


def getAvgRGB(image) -> (float, float, float):
    # Wczytanie obrazu
    img = image

    # Obliczenie średnich wartości natężenia RGB
    avg_r = img[:, :, 0].mean()
    avg_g = img[:, :, 1].mean()
    avg_b = img[:, :, 2].mean()

    colors = ['r', 'g', 'b']

    # Tworzenie wykresu
    plt.bar(['Red', 'Green', 'Blue'], [avg_r, avg_g, avg_b], color=colors)
    plt.xlabel('Color channel')
    plt.ylabel('Average value')
    plt.title('Average color values')
    plt.show()

    # Wyświetlenie wyników
    print(f"Średnie natężenie RGB: R={avg_r:.2f}, G={avg_g:.2f}, B={avg_b:.2f}")
    return avg_r, avg_g, avg_b

def getProportion(image):
    height, width, _ = image.shape
    return width / height

def getBanknoteValue(image):
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
    width, height = image.size

    if height > width:
        image = image.rotate(90, expand=True)
    
    return image

def getEntropy(image):
    return shannon_entropy(image)

def getVariance(image):
    return ndimage.variance(image)

def getSkewness(image):
    flatten_image = image.flatten()
    
    return skew(flatten_image)

# imagePath1 = '/Users/adamludwiczak/PycharmProjects/AnalizaDanych/CashRecognizer/Banknotes/Poland_10/38.jpg'
# imagePath2 = '/Users/adamludwiczak/PycharmProjects/AnalizaDanych/CashRecognizer/Banknotes/Poland_20/1_front.jpg'
# imagePath3 = '/Users/adamludwiczak/PycharmProjects/AnalizaDanych/CashRecognizer/Banknotes/Euro_5/1-0.jpg'
# image1 = getBanknote(imagePath1)
# prepareHistogram(image1)
# (r,g,b) = getAvgRGB(image1)
#
# image2 = getBanknote(imagePath2)
# prepareHistogram(image2)
# (r,g,b) = getAvgRGB(image2)
#
# image3 = getBanknote(imagePath3)
# prepareHistogram(image3)
# (r,g,b) = getAvgRGB(image3)