import json
from transformers import BertTokenizer

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# File path for the dataset
file_path = './en-sq.txt/OpenSubtitles.en-sq.en'

# Read lines
with open(file_path, 'r', encoding='utf-8') as en:
    english_sentences = en.readlines()

# Create contextual pairs
contextual_pairs = []
for i in range(len(english_sentences) - 1):
    input_sentence = english_sentences[i].strip()  # Current sentence
    response_sentence = english_sentences[i + 1].strip()  # Next sentence as response
    contextual_pairs.append((input_sentence, response_sentence))

#Save the tokenized data to a JSON file
output_file = "contextual_pairing.json"
with open(output_file, "w", encoding="utf-8") as f:
    print('printing tokenized data')
    json.dump(contextual_pairs, f, ensure_ascii=False, indent=4)

