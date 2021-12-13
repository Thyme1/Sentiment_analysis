import requests
from bs4 import BeautifulSoup
import re



# print(soup.prettify())  # ładniejszy sposób na wyświetlenie

# div class="comment-cloud__body relative"
# <p class="p-expanded js-expanded mb-0" style>
# mydivs = soup.find_all("div", {"class": "comment-cloud__body relative"})
for i in range(3):
    page_url = 'https://lubimyczytac.pl/ksiazki/opinie/' + str(i)
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # my_paragraphs = soup.find_all("p", class_="p-expanded js-expanded mb-0")
    my_ratings = soup.find_all("span", class_="big-number")
    for rating in my_ratings:
        rat = re.search(r'\d' , str(rating))
        print(rat.group())




    # with open("C:\\Users\\konra\\PycharmProjects\\Sentiment_analysis\\scraped.txt", "a") as fp:
    #     print([x.decompose() for x in soup.find_all("span", class_="big-number")])









