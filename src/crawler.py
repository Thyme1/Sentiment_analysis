import requests
from bs4 import BeautifulSoup
import re
import sys

original_stdout = sys.stdout
with open("src\\data.txt", "a", encoding="utf-8") as fp:
    for i in range(1040, 1120):
        page_url = 'https://lubimyczytac.pl/ksiazki/opinie/' + str(i)
        page = requests.get(page_url)
        soup = BeautifulSoup(page.content, 'html.parser')

        # with open('C:\\Users\\konra\\PycharmProjects\\Sentiment_analysis\\src\\html.html', 'w',encoding="utf-8") as ht:
        #     sys.stdout = ht
        #     print(soup.prettify())
        # <span class="rating-value">

        my_divs = soup.find_all("div", class_="col-12 mt-sm-3 mt-n4 col-md-9")
        while my_divs:
            current_div = my_divs.pop()

            soup = current_div
            my_ratings = soup.find_all("span", class_="big-number")
            my_paragraphs = soup.find_all("p", class_="p-expanded js-expanded mb-0")

            rating = re.search(r'\d+', str(my_ratings))

            my_paragraph_str = str(my_paragraphs)[1:-1]
            my_paragraph_str = my_paragraph_str.replace("\n", " ")
            my_paragraph_str = re.sub(r'</br>|<br>|<br/>', '', my_paragraph_str)

            paragraph = re.search(
                r'(?<=<p class=\"p-expanded js-expanded mb-0\" style=\"display:none;\"> )(.*?)(?=</p>)',
                my_paragraph_str)
            sys.stdout = fp
            if paragraph is None or rating is None:
                continue
            else:
                print(rating.group(), ';', paragraph.group(), sep='')
