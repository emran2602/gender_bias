import json
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


# Read the JSON file
with open('questions_matchinfo.json') as json_file:
    data = json.load(json_file)

# Separate questions based on gender
male_questions = []
female_questions = []

for entry in data.values():
    print(entry)
    gender = entry['gender']
    questions = entry['questions']

    tokenized_questions = [word_tokenize(question.lower()) for question in questions]


    if gender == 'M':
        male_questions.extend(tokenized_questions)
    elif gender == 'F':
        female_questions.extend(tokenized_questions)

# Write questions to separate text files
with open('male_questions.txt', 'w') as male_file:
    for question in male_questions:
        male_file.write(' '.join(question) + '\n')

with open('female_questions2.txt', 'w') as female_file:
    for question in female_questions:
        female_file.write(' '.join(question) + '\n')