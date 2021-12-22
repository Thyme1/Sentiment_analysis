import re

from sklearn.feature_extraction.text import TfidfVectorizer
from vowpalwabbit import pyvw
from sklearn.metrics import mean_squared_error


def change_into_training_data(opinion, test):
    train_line = ''
    opinion = opinion.split(';')
    if test:
        train_line += ' |' + 'characters:.' + str(len(opinion[1]))
    else:
        train_line += opinion[0] + ' |' + 'characters:.' + str(len(opinion[1]))
    word_list = opinion[1].split()
    train_line += ' words:.' + str(len(word_list))
    exclamation_marks = opinion[1].count('!')
    train_line += ' exclamation_marks:.' + str(exclamation_marks)
    positive_words_count = sum(el.lower() in POSITIVE for el in word_list)
    negative_words_count = sum(el.lower() in NEGATIVE for el in word_list)
    train_line += ' positive_words:.' + str(positive_words_count)
    train_line += ' negative_words:.' + str(negative_words_count)
    result = re.sub(r'[0-9]+|:', ' ', opinion[1])
    train_line += ' plain_text:.' + str(result.lower())

    return train_line


model = pyvw.vw(quiet=False)

POSITIVE = ['akcja', 'aktywna', 'aktywnie', 'aktywny', 'altruistyczny', 'ambitne', 'ambitnie', 'ambitny', 'anielska',
            'anielski', 'anielskie', 'anielsko', 'atrakcyjna', 'atrakcyjne', 'atrakcyjnie', 'atrakcyjny', 'autentyczna',
            'autentycznie', 'autentyczny', 'bezpośredni', 'bezpośrednia', 'bezpośrednie', 'bezpośrednio', 'bezstronna',
            'bezstronnie', 'bezstronny', 'bliska', 'bliski', 'bliskie', 'blisko', 'bliskość', 'bogactwo', 'bogata',
            'bogaty', 'bogini', 'boska', 'boski', 'bratnia dusza', 'brawo', 'buziaki', 'bystra', 'bystre', 'bystry',
            'błoga', 'błogi', 'błogo', 'błogosławieństwa', 'błogosławieństwo', 'błogosławiony', 'błogość', 'błyszczy',
            'cały', 'całym sercem', 'ceniona', 'ceniony', 'cenna', 'cenny', 'charyzma', 'charyzmatyczna',
            'charyzmatyczny', 'chętna', 'chętnie', 'chętny', 'chęćciepły', 'ciekawe', 'ciekawie', 'ciepła', 'ciepło',
            'cierpliwość', 'cierpliwy', 'cieszyć', 'cieszę', 'cnotliwy', 'cud', 'cudowny', 'czarujący', 'czule',
            'czuć się dobrze', 'czuła', 'czułość', 'czuły', 'czysta miłość i światło', 'czysty', 'delikatna',
            'delikatnie', 'delikatny', 'dobra', 'dobre', 'dobrze', 'empatyczna', 'empatyczny', 'fantastyczna',
            'fantastyczne', 'fantastycznie', 'fantastyczny', 'fantazja', 'fantazje', 'geniusz', 'godny podziwu',
            'gorliwa', 'gorliwie', 'gorliwy', 'gratulacje', 'gratuluję', 'harmonia', 'harmonijnie', 'harmonijny',
            'hojna', 'hojne', 'hojnie', 'hojny', 'humor', 'humorystycznie', 'humorystyczny', 'huumorystyczna',
            'idealna', 'idealnie', 'ideał', 'imponuje', 'imponująca', 'imponujące', 'imponujący', 'innowacja',
            'innowacyjna', 'innowacyjnie', 'innowacyjny', 'inspiracja', 'inspirować', 'inspirująca', 'inspirujące',
            'instynktownie', 'intelegtualny', 'intelekt', 'intelektualna', 'inteligencja', 'inteligentna',
            'inteligentny', 'intuicja', 'intuicyjna', 'intuicyjny', 'istotna', 'istotnie', 'istotny', 'jasna', 'jasne',
            'kochający', 'kocham', 'kochana', 'kochałam', 'kochałem', 'komfort', 'komfortowa', 'komfortowe',
            'korzystna', 'korzystne', 'korzystnie', 'kreatywna', 'kreatywne', 'kreatywnie', 'kreatywny', 'kwitnący',
            'lekka', 'lekki', 'lekko', 'mistrz', 'mistrzostwo', 'mistrzowski', 'mistrzyni', 'mądra', 'mądre', 'mądrość',
            'mądry', 'mądrze', 'młodzieńczy', 'najlepiej', 'najlepsza', 'najlepsze', 'najlepszy', 'niebybywały',
            'niebywałacałuski', 'niebywałe', 'niezapomniana', 'niezapomniana', 'niezapomniany', 'niezapomniany',
            'niezawodna', 'niezawodnie', 'niezawodny', 'niezrównana', 'niezrównany', 'niezwykla', 'niezwykły',
            'niezywkła', 'oczarowana', 'oczarowany', 'oryginalnie', 'oryginalność', 'oryginalny', 'oszałamiająca',
            'oszałamiające', 'oszałamiający', 'pełen', 'pełna', 'podekscytowana', 'podekscytowanie', 'podekscytowany',
            'pokochałam', 'pokochałem', 'pomysłowa', 'pomysłowy', 'pozytwna', 'pozytywnie', 'przygoda', 'przyjemna',
            'przyjemne', 'przyjemnie', 'przyjemny', 'radosna', 'radosne', 'radosne', 'radośnie', 'rozkosz', 'rozkoszna',
            'rozkosznie', 'rozkoszny', 'silna', 'silnie', 'silny', 'sukces', 'szczera', 'szczerość', 'szczery',
            'szczerze', 'szlachetna', 'szlachetnie', 'szlachetny', 'sztuka', 'twórcza', 'twórcze', 'twórczy', 'udana',
            'udanie', 'ukochana', 'ukochany', 'ulubiona', 'ulubione', 'urok', 'uwielbiam', 'wciągająca', 'wciągnął',
            'wciągnęła', 'wow', 'wyobraźnia', 'wyobraźnię', 'zabawa', 'zabawna', 'zabawnie', 'zachwycona', 'zachwycony',
            'zachwyt', 'zywa', 'ładna', 'ładnie', 'ładny', 'łał', 'świadoma', 'świadomie', 'świadomy', 'świetna',
            'świetne', "😉", "każdy", 'świetnie', "esencja", "emanowała", "rzetelna", "rzetelnie", "rzetelność",
            "czekam", 'świetnyciekawa', 'żywo', 'żywy', "świetna", "świetny", "świetnie", "polecam", "gorąco",
            "doskonała", "doskonale", "doskonały", "porusza", "poruszyła", 'poruszająca', "zaciekawiła",
            "zaintrygowała", "zafascynowała", "zafascynowanie", "zaciekawienie", "zaciekawiony", "zaciekawiona",
            "interesująca", "bardzo", "fajna", "fajny", "fajnie", "❤"]

NEGATIVE = ["najgorsza", "najgorzej", "najgorszy", "słaba", "słaby", "słabo" "porażka", "zła", "nudna", "nudno",
            "nudnie",
            "najnudniejsza", "najnudniejszy", "nuda", "nieciekawa", "nieciekawy", "nieciekawe", "nieciekawie",
            "niedobra", "niedobry",
            "niedobrze", "mieszane", "żałuję", "niby", "spadek","okropna","okropnie","okropny","nierealistyczne","nierealistyczna","nierealistycznie" ]
print(len(POSITIVE))
print(len(NEGATIVE))
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
