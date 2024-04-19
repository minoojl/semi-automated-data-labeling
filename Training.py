import pandas as pd
import random
import re

# Assume 'text_corpus' is a large string containing the entire text corpus.
text_corpus = "..."  # Your text corpus goes here
N = 5  # The number of sentences to select after each question
L = 3  # The number of human labelers

# Function to grep all questions from the corpus
def grep_questions(corpus):
    # Regex to match questions. This is a simple pattern and might need to be adapted
    question_pattern = r'\?'
    questions = re.findall(question_pattern, corpus)
    return questions

# Function to select N following sentences from the corpus
def select_n_following_sentences(question, corpus, n):
    # Split the corpus into sentences
    sentences = corpus.split('.')
    # Find the index of the question
    question_idx = sentences.index(question)
    # Select N following sentences
    following_sentences = sentences[question_idx + 1 : question_idx + 1 + n]
    return following_sentences

# Create dataset from questions and their following sentences
def create_dataset(questions, corpus, n):
    dataset = []
    for question in questions:
        following_sentences = select_n_following_sentences(question, corpus, n)
        dataset.append({'question': question, **{f'answer{i+1}': ans for i, ans in enumerate(following_sentences)}})
    return pd.DataFrame(dataset)

# Function to manually label the dataset
def manually_label(dataset, L, Q):
    labeled_data = pd.DataFrame()
    for i in range(L):
        # Select Q random records
        random_records = dataset.sample(n=Q)
        # Here you would implement your actual labeling logic
        # For example, calling an external API or presenting the data to a human through a UI
        # We will just copy the data to simulate labeling
        labeled_data = labeled_data.append(random_records, ignore_index=True)
    return labeled_data

# Function to add machine labeled data to human labeled data
def combine_human_machine_labels(human_labeled, machine_labeled):
    combined = human_labeled.append(machine_labeled, ignore_index=True)
    return combined

# Main process
questions = grep_questions(text_corpus)
dataset = create_dataset(questions, text_corpus, N)

# Simulate machine labeling process
# In a real-world scenario, you would use a machine learning model to label the data
machine_labeled = dataset.copy()  # This is just a placeholder

# Manually label Q random records by L labelers
Q = 10  # Number of records to label
human_labeled = manually_label(dataset, L, Q)

# Add records from the machine table to the human table
combined_labeled_data = combine_human_machine_labels(human_labeled, machine_labeled)

# Return the labeled dataset
print(combined_labeled_data.head())

# Save the dataset to a CSV file
combined_labeled_data.to_csv('labeled_dataset.csv', index=False)
