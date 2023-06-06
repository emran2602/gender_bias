import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Read the JSON file
with open('questions_matchinfo.json') as json_file:
    data = json.load(json_file)

# Separate questions based on gender
male_questions = []
female_questions = []

# Get the list of English stopwords and punctuation
stopwords_list = set(stopwords.words('english'))
punctuation_list = set(string.punctuation)

for entry in data.values():
    gender = entry.get('gender')
    questions = entry.get('questions')

    # Preprocess each question
    preprocessed_questions = []
    for question in questions:
        # Tokenize the question
        tokens = word_tokenize(question)

        # Convert tokens to lowercase
        tokens_lower = [token.lower() for token in tokens]

        # Remove punctuation
        tokens_no_punct = [token for token in tokens_lower if token not in punctuation_list]

        # Remove stopwords
        tokens_no_stopwords = [token for token in tokens_no_punct if token not in stopwords_list]

        # Add preprocessed question to the list
        preprocessed_questions.append(tokens_no_stopwords)

    if gender == 'M':
        male_questions.extend(preprocessed_questions)
    elif gender == 'F':
        female_questions.extend(preprocessed_questions)

# Write questions to separate text files
with open('male_questions2.txt', 'w') as male_file:
    for question in male_questions:
        male_file.write(' '.join(question) + '\n')

with open('female_questions2.txt', 'w') as female_file:
    for question in female_questions:
        female_file.write(' '.join(question) + '\n')