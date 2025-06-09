import requests as r
from bs4 import BeautifulSoup

res = r.get("https://de.wikipedia.org/w/index.php?title=Liste_von_Abk%C3%BCrzungen_im_Eisenbahnwesen&oldid=256152030").text

soup = BeautifulSoup(res, 'lxml')

tables = soup.find_all('table')

