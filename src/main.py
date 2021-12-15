import requests
from bs4 import BeautifulSoup
import re
import sys


with open("C:\\Users\\konra\\PycharmProjects\\Sentiment_analysis\\scraped.txt", "a",encoding="utf-8") as fp:
    for i in range(10):
        page_url = 'https://lubimyczytac.pl/ksiazki/opinie/' + str(i)
        page = requests.get(page_url)
        soup = BeautifulSoup(page.content, 'html.parser')

        my_ratings = soup.find_all("span", class_="big-number")
        my_paragraphs = soup.find_all("p", class_="p-expanded js-expanded mb-0")
        my_paragraph_str = str(my_paragraphs)[1:-1]
        my_paragraph_str = my_paragraph_str.replace("\n", " ")
        my_paragraph_str = my_paragraph_str.replace("<br/>", "")
        my_paragraph_str = my_paragraph_str.replace("<br>", "")
        my_paragraph_str = re.sub(r'</br>', '', my_paragraph_str)
        paragraph = re.findall(r'(?<=<p class=\"p-expanded js-expanded mb-0\" style=\"display:none;\"> )(.*?)(?=</p>)',my_paragraph_str)



        my_ratings = soup.find_all("span", class_="big-number")
        for rating,item in zip(my_ratings,paragraph):
            rat = re.search(r'\d+', str(rating))
            sys.stdout = fp
            print(rat.group(),';',item,sep='')








