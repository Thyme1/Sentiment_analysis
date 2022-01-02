import re
import sys

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
    train_line += ' exclamation_marks:.' + str(opinion[1].count('!'))
    train_line += ' question_marks:.' + str(opinion[1].count('?'))
    result2 = [re.sub(r'[:.,?!;"„”]', ' ', i) for i in word_list]
    positive_words_count = sum(el.lower() in POSITIVE for el in result2)
    negative_words_count = sum(el.lower() in NEGATIVE for el in result2)
    train_line += ' positive_words:.' + str(positive_words_count)
    train_line += ' negative_words:.' + str(negative_words_count)
    result = re.sub(r'[0-9]+|[:.,?!;"„”]', ' ', opinion[1])
    train_line += ' plain_text:.' + str(result.lower())


    return train_line


model = pyvw.vw(quiet=False)

POSITIVE = ['akcja', 'aktywna', 'aktywnie', 'aktywny', 'altruistyczny', 'ambitne', 'ambitnie', 'ambitny', 'anielska',
            'anielski', 'anielskie', 'anielsko', 'atrakcyjna', 'atrakcyjne', 'atrakcyjnie', 'atrakcyjny', 'autentyczna',
            'autentycznie', 'autentyczny', 'bardzo', 'bawić', 'bezpośredni', 'bezpośrednia', 'bezpośrednie',
            'bezpośrednio', 'bezstronna', 'bezstronnie', 'bezstronny', 'bliska', 'bliski', 'bliskie', 'blisko',
            'bliskość', 'bogactwo', 'bogata', 'bogaty', 'bogini', 'bomba', 'boska', 'boski', 'bratnia dusza', 'brawo',
            'brawo', 'buziaki', 'bystra', 'bystre', 'bystry', 'błoga', 'błogi', 'błogo', 'błogosławieństwa',
            'błogosławieństwo', 'błogosławiony', 'błogość', 'błyszczy', 'cały', 'całym sercem', 'ceniona', 'ceniony',
            'cenna', 'cenny', 'charyzma', 'charyzmatyczna', 'charyzmatyczny', 'chętna', 'chętnie', 'chętny',
            'chęćciepły', 'ciekawe', 'ciekawie', 'ciepła', 'ciepło', 'cierpliwość', 'cierpliwy', 'cieszyć', 'cieszę',
            'cnotliwy', 'cud', 'cudowny', 'cudowna','cudowne','cudownie','cudna','cudnie','cudne','cudny' 'czarujący', 'czekam', 'czule', 'czuć się dobrze', 'czuła', 'czułość',
            'czuły', 'czysta miłość i światło', 'czysty', 'delikatna', 'delikatnie', 'delikatny', 'dobra', 'dobre',
            'dobrze', 'doskonale', 'doskonała', 'doskonały', 'dowcipna', 'emanowała', 'empatyczna', 'empatyczny',
            'esencja', 'fajna', 'fajnie', 'fajny', 'fantastyczna', 'fantastyczne', 'fantastycznie', 'fantastyczny',
            'fantazja', 'fantazje', 'geniusz', 'godny podziwu', 'gorliwa', 'gorliwie', 'gorliwy', 'gorąco',
            'gratulacje', 'gratuluję', 'harmonia', 'harmonijnie', 'harmonijny', 'hojna', 'hojne', 'hojnie', 'hojny',
            'homor', 'humor', 'humor', 'humorystycznie', 'humorystyczny', 'huumorystyczna', 'idealna', 'idealna',
            'idealnie', 'ideał', 'imponuje', 'imponująca', 'imponujące', 'imponujący', 'innowacja', 'innowacyjna',
            'innowacyjnie', 'innowacyjny', 'inspiracja', 'inspirować', 'inspirująca', 'inspirujące', 'instynktownie',
            'intelegtualny', 'intelekt', 'intelektualna', 'inteligencja', 'inteligentna', 'inteligentny',
            'interesująca', 'intuicja', 'intuicyjna', 'intuicyjny', 'istotna', 'istotnie', 'istotny', 'jasna', 'jasne',
            'każdy', 'klimat', 'klimatem', 'klimatu', 'kochający', 'kocham', 'kochana', 'kochałam', 'kochałem',
            'komfort', 'komfortowa', 'komfortowe', 'korzystna', 'korzystne', 'korzystnie', 'kreatywna', 'kreatywne',
            'kreatywnie', 'kreatywny', 'kwitnący', 'lekka', 'lekki', 'lekko', 'mega', 'mistrz', 'mistrzostwo',
            'mistrzowski', 'mistrzyni', 'mądra', 'mądre', 'mądrość', 'mądry', 'mądrze', 'młodzieńczy', 'najlepiej',
            'najlepsza', 'najlepsze', 'najlepszy', 'napięcie', 'napięciu', 'niebybywały', 'niebywałacałuski',
            'niebywałe', 'niezapomniana', 'niezapomniana', 'niezapomniany', 'niezapomniany', 'niezawodna',
            'niezawodnie', 'niezawodny', 'niezrównana', 'niezrównany', 'niezwykla', 'niezwykły', 'niezywkła',
            'oczarowana', 'oczarowany', 'oryginalnie', 'oryginalność', 'oryginalny', 'oszałamiająca', 'oszałamiające',
            'oszałamiający', 'pełen', 'pełna', 'pochłonęła', 'podekscytowana', 'podekscytowanie', 'podekscytowany',
            'pokochałam', 'pokochałem', 'polecam', 'pomysłowa', 'pomysłowy', 'porusza', 'poruszająca', 'poruszyła',
            'potężny', 'pozytwna', 'pozytywnie', 'przygoda', 'przyjemna', 'przyjemne', 'przyjemnie', 'przyjemnie',
            'przyjemny', 'radosna', 'radosne', 'radosne', 'radośnie', 'rozkosz', 'rozkoszna', 'rozkosznie', 'rozkoszny',
            'rzetelna', 'rzetelnie', 'rzetelność', 'silna', 'silnie', 'silny', 'sukces', 'szczera', 'szczerość',
            'szczery', 'szczerze', 'szlachetna', 'szlachetnie', 'szlachetny', 'sztuka', 'twórcza', 'twórcze', 'twórczy',
            'udana', 'udanie', 'ukochana', 'ukochany', 'ulubiona', 'ulubione', 'urok', 'uwielbiam', 'uśmiech', 'warto',
            'wartościowa', 'wartościowe', 'wciągająca', 'wciągnął', 'wciągnęła', 'wow', 'wyobraźnia', 'wyobraźnię',
            'zabawa', 'zabawa', 'zabawna', 'zabawnie', 'zachwycona', 'zachwycony', 'zachwyt', 'zaciekawienie',
            'zaciekawiona', 'zaciekawiony', 'zaciekawiła', 'zafascynowanie', 'zafascynowała', 'zaintrygowała', 'zywa',
            'ładna', 'ładnie', 'ładny', 'łał', 'świadoma', 'świadomie', 'świadomy', 'świetna', 'świetna', 'świetne',
            'świetnie', 'świetnie', 'świetny', 'świetnyciekawa', 'żywo', 'żywy', '❤', '🌟', '🔥', '😃', '😅', '😉']

NEGATIVE = ['bez wyrazu', 'brak', 'brakuje', 'drażni', 'drażniła', 'drażniły', 'głupia', 'głupio', 'głupią', 'kicz',
            'kiczowata', 'mało', 'mieszane', 'minus', 'minusem', 'męczyłam', 'męczyłem', 'męcząca', 'męczące',
            'męczący', 'mękazbędna', 'najgorsza', 'najgorszy', 'najgorzej', 'najnudniejsza', 'najnudniejszy', 'niby',
            'nie podoba', 'nie podobała', 'nie porwał', 'nie porwała', 'nie trafia', 'nie trafiła', 'niechętna',
            'niechętnie', 'nieciekawa', 'nieciekawe', 'nieciekawie', 'nieciekawy', 'niedobra', 'niedobry', 'niedobrze',
            'nierealistyczna', 'nierealistyczne', 'nierealistycznie', 'niespójna', 'niespójna', 'niespójne',
            'niespójnie', 'niespójny', 'nijacy', 'nijak', 'nijaka', 'nijaki', 'nijako', 'nuda', 'nudna', 'nudnie',
            'nudno', 'nużąca', 'nużące', 'nużący', 'odradzam', 'okropna', 'okropnie', 'okropny', 'ostrzegam',
            'ostrzeżenie', 'powtarza', 'powtarzalna', 'powtarzalny', 'pretensjonalna', 'pretensjonalną', 'problem',
            'przykro', 'płytka', 'płytki', 'płytkie', 'spadek', 'szajs', 'słaba', 'słabiutko', 'słaboporażka', 'słaby',
            'trudno', 'z trudem', 'zła', 'żałuję', '☹',"nie", "najgorsze"]

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
