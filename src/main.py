import re

from sklearn.feature_extraction.text import TfidfVectorizer
from vowpalwabbit import pyvw
from sklearn.metrics import mean_squared_error

vectorizer = TfidfVectorizer()


def change_into_training_data(opinion, test):
    train_line = ''
    opinion = opinion.split(';')
    if test:
        train_line += '| ' + 'characters:.' + str(len(opinion[1]))
    else:
        train_line += opinion[0] + ' | ' + 'characters:.' + str(len(opinion[1]))
    word_list = opinion[1].split()
    train_line += ' words:.' + str(len(word_list))
    exclamation_marks = opinion[1].count('!')
    train_line += ' exclamation_marks:.' + str(exclamation_marks)
    positive_words_count = sum(el in POSITIVE for el in word_list)
    negative_words_count = sum(el in NEGATIVE for el in word_list)
    train_line += ' positive_words:.' + str(positive_words_count)
    train_line += ' negative_words:.' + str(negative_words_count)
    result = re.sub(r'[0-9]+|:', ' ', opinion[1])
    train_line += ' plain_text:.' + str(result.lower())
    return train_line


model = pyvw.vw(quiet=False)

POSITIVE = ['najlepsza', "dobra", "dobrze", "świetna", "świetnie" "ciekawa", "ciekawe" "ulubiona", "ulubione",
            "wciągnęła", "wciągająca", "przyjemna", "przyjemnie", "udana"]  # odmiana słów - regex?
NEGATIVE = ["najgorsza", "słaba", "porażka", "zła", "nudna", "nieciekawa", "niedobra"]

train_examples = []

with open("src\\train_data.txt", "r", encoding="utf-8") as train_data:
    lines = train_data.readlines()
    for opinion in lines:
        train_line = change_into_training_data(opinion, False)
        train_examples.append(train_line)

for example in train_examples:
    model.learn(example)
model.finish()

with open("src\\test_data.txt", "r", encoding="utf-8") as train_data:
    lines = train_data.readlines()
    test_set = []
    for opinion in lines:
        test_line = change_into_training_data(opinion, True)
        opinion = opinion.split(';')
        test_set.append([opinion[0], test_line])

original_rating = []
predicted_rating = []
for i in range(len(test_set)):
    prediction = model.predict(test_set[i][1])
    original_rating.append(test_set[i][0])
    predicted_rating.append(prediction)
print("rmse: " + str(mean_squared_error(original_rating, predicted_rating, squared=False)))
