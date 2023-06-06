import json
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Open the JSON file
with open('text_commentaries.json', 'r') as json_file:
    data = json.load(json_file)

# Extract and preprocess the commentary
commentary_list = []
for item in data:
    commentary = item['commentary']
    commentary = commentary.lower()  # Convert to lowercase
    tokens = word_tokenize(commentary)  # Tokenize into words
    commentary_list.append(' '.join(tokens))  # Join tokens back into a sentence

# Save the preprocessed commentary to a text file
with open('commentary.txt', 'w') as txt_file:
    txt_file.write('\n'.join(commentary_list))