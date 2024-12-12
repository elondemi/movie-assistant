import json
from transformers import BertTokenizer

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# File path for the dataset
file_path = './en-sq.txt/OpenSubtitles.en-sq.en'

# List to store tokenized results
tokenized_data = []

# Open and read the file line by line
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()  # Reads all lines into a list

# Tokenize each line and store the results
for line in lines[:20]:
    line = line.strip()  # Remove leading/trailing spaces or newlines
    if line:  # Ensure the line is not empty
        tokens = tokenizer.tokenize(line)  # Tokenize the text
        input_ids = tokenizer.encode(line,  add_special_tokens=True)  # Convert to token IDs

        # Save the results in a dictionary
        tokenized_data.append({
            "original_text": line,
            "tokens": tokens,
            "input_ids": input_ ids
        })

# Save the tokenized data to a JSON file
output_file = "tokenized_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    print('printing tokenized data')
    json.dump(tokenized_data, f, ensure_ascii=False, indent=4)

print(f"Tokenized data saved to {output_file}")
