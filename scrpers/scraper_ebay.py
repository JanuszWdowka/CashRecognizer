import requests
from bs4 import BeautifulSoup
import os

# link do strony z wynikami wyszukiwania aukcji
url = 'https://www.ebay.com/sch/i.html?_from=R40&_trksid=p2334524.m570.l1313&_nkw=1+Dollar&_sacat=0&LH_TitleDesc=0&_odkw=1+Dollar+Banknote&_osacat=0&_ipg=240'

# wysyłanie zapytania HTTP do serwera i pobieranie zawartości strony
response = requests.get(url)
html = response.content

# parsowanie strony HTML za pomocą BeautifulSoup
soup = BeautifulSoup(html, 'html.parser')

# wyszukiwanie linków do aukcji
auction_links = []
for a in soup.find_all('a', {'class': 's-item__link'}):
    auction_links.append(a.get('href'))

print(len(auction_links))
# pobieranie zdjęć z każdej aukcji  1q1 q
for i, link in enumerate(auction_links):
    response = requests.get(link)
    html = response.content
    soup = BeautifulSoup(html, 'html.parser')

    # wyszukiwanie tagów <img> i pobieranie adresów URL zdjęć
    image_urls = []
    for img in soup.find_all('img'):
        image_url = img.get('src')
        if image_url is not None and (image_url.endswith('.jpg') or image_url.endswith('.jpeg') or image_url.endswith('.png')):
            image_urls.append(image_url)
    print(len(image_urls))
    # pobieranie zdjęć i zapisywanie ich do katalogu
    for j, url in enumerate(image_urls):
        response = requests.get(url)
        with open(f'../Banknotes/USA_1/{i}-{j}.jpg', 'wb') as f:
            f.write(response.content)