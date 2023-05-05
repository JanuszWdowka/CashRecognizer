import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import sobel
from skimage.morphology import binary_closing, disk
from skimage.measure import label, regionprops
import skimage


def getBanknote(imagePath):
    # wczytaj obraz banknotu
    image = io.imread(imagePath)

    # przetwarzanie wstępne - zamiana na obraz w skali szarości i wygładzenie
    gray_image = skimage.color.rgb2gray(image)
    blurred_image = skimage.filters.gaussian(gray_image, sigma=2.0)

    # wykrywanie krawędzi przy użyciu operatora Sobela
    edge_sobel = sobel(blurred_image)

    # binaryzacja i zamknięcie - przekształcenia morfologiczne służące do usunięcia małych artefaktów
    thresh = skimage.filters.threshold_otsu(edge_sobel)
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