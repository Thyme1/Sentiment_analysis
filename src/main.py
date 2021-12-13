import requests
from bs4 import BeautifulSoup

page_url = 'https://lubimyczytac.pl/ksiazki/opinie'
page = requests.get(page_url)

soup = BeautifulSoup(page.content, 'html.parser')

# print(soup.prettify())  # ładniejszy sposób na wyświetlenie

# div class="comment-cloud__body relative"
# <p class="p-expanded js-expanded mb-0" style>
# mydivs = soup.find_all("div", {"class": "comment-cloud__body relative"})
my_paragraphs = soup.find_all("p", class_="p-expanded js-expanded mb-0")
my_ratings = soup.find_all("span", class_="big-number")

print(my_ratings)




