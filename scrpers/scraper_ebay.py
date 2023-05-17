"""
Klasa pomocnicza posiadająca skrypt do pobierania zdjęć banknotów z Ebaya dla dostarczenia modelowi zdjęć na podstawie,
których będzie się uczył. Zdjęcia są zapisywane lokalnie na dysku.
"""
import requests
from bs4 import BeautifulSoup

url = 'https://www.ebay.com/sch/i.html?_from=R40&_nkw=50+pounds&_sacat=0&_ipg=240'

response = requests.get(url)
html = response.content

soup = BeautifulSoup(html, 'html.parser')

auction_links = []
for a in soup.find_all('a', {'class': 's-item__link'}):
    auction_links.append(a.get('href'))


for i, link in enumerate(auction_links):
    response = requests.get(link)
    html = response.content
    soup = BeautifulSoup(html, 'html.parser')

    image_urls = []
    divs_with_tak_class = soup.find_all('div', {'class': 'ux-image-carousel-item'})
    for div in divs_with_tak_class:
        img = div.find('img')
        image_url = img.get('src')
        image_url_data = img.get('data-src')
        if image_url is not None and (image_url.endswith('.jpg') or image_url.endswith('.jpeg') or image_url.endswith('.png')):
            image_urls.append(image_url)
        if image_url_data is not None and (image_url_data.endswith('.jpg') or image_url_data.endswith('.jpeg') or image_url_data.endswith('.png')):
            image_urls.append(image_url_data)

    for j, url in enumerate(image_urls):
        response = requests.get(url)
        with open(f'../Banknotes/UK_50/{i}-{j}.jpg', 'wb') as f:
            f.write(response.content)