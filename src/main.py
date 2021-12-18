from vowpalwabbit import pyvw

model = pyvw.vw(quiet=True)

POSITIVE = ['najlepsza', "dobra", "dobrze", "świetna", "świetnie" "ciekawa", "ciekawe" "ulubiona", "ulubione", "wciągnęła", "wciągająca", "przyjemna", "przyjemnie", "udana"]  # odmiana słów - regex?
NEGATIVE = ["najgorsza", "słaba", "porażka" ]

train_examples = []

with open("src\\placeholder_data.txt", "r", encoding="utf-8") as train_data:
    lines = train_data.readlines()
    for opinion in lines:
        train_line = ''
        opinion = opinion.split(';')
        train_line += opinion[0] + ' | ' + 'characters:.' + str(len(opinion[1]))
        word_list = opinion[1].split()
        train_line += ' words:.' + str(len(word_list))
        exclamation_marks = opinion[1].count('!')
        train_line += ' exclamation_marks:.' + str(exclamation_marks)
        positive_words_count = sum(el in POSITIVE for el in word_list)
        negative_words_count = sum(el in NEGATIVE for el in word_list)
        train_line += ' positive_words:.' + str(positive_words_count)
        train_line += ' negative_words:.' + str(negative_words_count)

        train_examples.append(train_line)

    print(train_examples)
