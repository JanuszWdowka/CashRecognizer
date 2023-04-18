import requests
from bs4 import BeautifulSoup
import os

url = 'https://www.ebay.com/sch/i.html?_from=R40&_trksid=p2334524.m570.l1313&_nkw=20+pounds+uk&_sacat=0&LH_TitleDesc=0&_odkw=10+pounds+uk&_osacat=0&_ipg=240'

response = requests.get(url)
html = response.content

soup = BeautifulSoup(html, 'html.parser')

auction_links = []
for a in soup.find_all('a', {'class': 's-item__link'}):
    auction_links.append(a.get('href'))

print(len(auction_links))

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
        with open(f'../Banknotes/UK_20/{i}-{j}.jpg', 'wb') as f:
            f.write(response.content)