import csv
import os
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translation_pipeline = pipeline("translation_en_to_de", model="facebook/bart-large-cnn")

def summarize_text(text, max_len=30, min_len=10):
    max_sequence_length = 1024  # Maximum sequence length for the model
    if len(text) > max_sequence_length:
        text = text[:max_sequence_length]

    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
    return summary

def translate_en_to_de(text):
    german_text = translation_pipeline(text)[0]['translation_text']
    return german_text


# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the data folder
data_folder = os.path.join(current_dir, '..', 'data')

file_path = os.path.join(data_folder, 'bertopic_result.csv')

# # Load data from the CSV file
# articles = []
# with open(file_path, 'r', encoding='utf-8') as file:
#     reader = csv.DictReader(file)
#     fieldnames = reader.fieldnames + ['summary']
#     for row in reader:
#         text = row['text']
#         summary = summarize_text(text)
#         row['summary'] = summary
#         articles.append(row)

# # Save the updated data to a new CSV file
# output_file_path = os.path.join(data_folder, 'final_data.csv')
# with open(output_file_path, 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(articles)
