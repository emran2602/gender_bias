import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize

# Read the commentaries from the text file
with open('commentary.txt', 'r') as file:
    commentaries = file.read()

# Split the commentaries into individual sentences
sentences = sent_tokenize(commentaries)

# Preprocess each sentence and save in the preprocessed file
preprocessed_file = 'preprocessed_commentaries.txt'
with open(preprocessed_file, 'w') as file:
    for sentence in sentences:
        # Tokenize the sentence into words
        tokens = word_tokenize(sentence)

        # Lowercase, remove punctuation, and remove stop words
        lowercase_tokens = [token.lower() for token in tokens]
        filtered_tokens = [token for token in lowercase_tokens if token.isalpha()] 

        # Save the preprocessed sentence in the file
        file.write(' '.join(filtered_tokens) + '\n')