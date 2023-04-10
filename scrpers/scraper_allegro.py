import time
import os
import random
import string
import urllib.request
from selenium import webdriver

def get_random_string(lenght):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(lenght))
    return result_str

i = 0
isNextPage = True
elements = list()
links = list()

driver = webdriver.Chrome('/Users/adamludwiczak/Downloads/chromedriver_mac_arm64/chromedriver')
driver.implicitly_wait(5)

driver.get('https://archiwum.allegro.pl/kategoria/polska-od-1994-13843?string=500z%C5%82%20banknot')
banknote_name = "images_500"
path_banknotes = f'../Banknotes/{banknote_name}'
time.sleep(5)

while isNextPage:
    try:
        x = driver.find_elements_by_css_selector("a.a607fda")
        for y in x:
            try:
                links.append(y.get_attribute("href"))
            except Exception:
                print("Zły link lub nie ma linku w znalezionym itemie")
            finally:
                pass
        next_page_link = driver.find_element_by_css_selector("a[aria-label='następna strona']")
        next_page_link.click()
        time.sleep(1)
    except Exception:
        print("Brak nastepnej strony")
        isNextPage = False
    finally:
        pass

print(f"Ilość poprawnych linków: {len(links)}")

if not os.path.exists(path_banknotes):
    os.mkdir(path_banknotes)

for link in links:
    try:
        driver.get(link)
        img_element = driver.find_element_by_css_selector("img[data-role='asi-gallery__image']")
        img_src = img_element.get_attribute("src")
        urllib.request.urlretrieve(img_src, f'{path_banknotes}/{get_random_string(10)}.jpg')
    except Exception:
        print("Chyba nie ma obrazka")
    finally:
        i = i + 1
        print(f"{i}/{len(links)}")
        pass


driver.quit()
