import os
import re
import torch
import pandas as pd
import pickle
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizerFast
from datasets.multi_task_dataset import MultiTaskDataset

# Load tokenizer globally to avoid redundant initialization
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Mapping NER labels to numerical indices
ner_label_map = {
    "O": 0,
    "B-art": 1, "I-art": 2,  # Artifact
    "B-eve": 3, "I-eve": 4,  # Event
    "B-geo": 5, "I-geo": 6,  # Geographical Entity
    "B-gpe": 7, "I-gpe": 8,  # Countries, Cities, States
    "B-nat": 9, "I-nat": 10,  # Natural Phenomena
    "B-org": 11, "I-org": 12,  # Organizations
    "B-per": 13, "I-per": 14,  # Persons
    "B-tim": 15, "I-tim": 16,  # Time Expressions
}

# Directory for storing processed datasets
PROCESSED_DIR = "./data/processed/"


def save_processed_data(data, filename):
    """Save processed dataset using pickle."""
    filepath = os.path.join(PROCESSED_DIR, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
    print(f"Processed data saved: {filepath}")


def load_processed_data(filename):
    """Load processed dataset from cache if available."""
    filepath = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded cached data: {filepath}")
        return data
    return None


def prepare_classification_data(file_path, batch_size=8, test_size=100, seed=42):
    """Loads and prepares classification dataset with caching."""
    cache_filename = "classification_dataset.pkl"

    # Load cached data if available
    cached_data = load_processed_data(cache_filename)
    if cached_data:
        classification_train_data, classification_test_data = cached_data
    else:
        # Load raw data from file
        classification_data = load_data_from_xlsx(file_path, "classification")
        train_size = len(classification_data) - test_size

        # Split dataset into training and test sets
        classification_train_data, classification_test_data = random_split(
            classification_data, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
        )

        # Cache processed data
        save_processed_data((classification_train_data, classification_test_data), cache_filename)

    # Create datasets
    classification_train_dataset = MultiTaskDataset(task="classification", data=classification_train_data, tokenizer=tokenizer)
    classification_test_dataset = MultiTaskDataset(task="classification", data=classification_test_data, tokenizer=tokenizer)

    # Create DataLoaders
    classification_train_loader = DataLoader(classification_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    classification_test_loader = DataLoader(classification_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return classification_train_loader, classification_test_loader, len(classification_train_dataset), len(classification_test_dataset)


def prepare_ner_data(file_path, batch_size=8, test_size=100, seed=42, ner_label_map=ner_label_map):
    """Loads and prepares NER dataset with caching."""
    cache_filename = "ner_dataset.pkl"

    # Load cached data if available
    cached_data = load_processed_data(cache_filename)
    if cached_data:
        ner_train_data, ner_test_data = cached_data
    else:
        # Load raw data from file
        ner_data = load_data_from_xlsx(file_path, "ner")
        train_size = len(ner_data) - test_size

        # Split dataset into training and test sets
        ner_train_data, ner_test_data = random_split(
            ner_data, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
        )

        # Cache processed data
        save_processed_data((ner_train_data, ner_test_data), cache_filename)

    # Create datasets
    ner_train_dataset = MultiTaskDataset(task="ner", data=ner_train_data, tokenizer=tokenizer, label_map=ner_label_map)
    ner_test_dataset = MultiTaskDataset(task="ner", data=ner_test_data, tokenizer=tokenizer, label_map=ner_label_map)

    # Create DataLoaders
    ner_train_loader = DataLoader(ner_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    ner_test_loader = DataLoader(ner_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return ner_train_loader, ner_test_loader, len(ner_train_dataset), len(ner_test_dataset)


def load_data_from_xlsx(file_path, task):
    """
    Load and preprocess dataset from an XLSX file.

    - Classification task expects 'review' and 'sentiment' columns.
    - NER task expects 'Sentence #', 'Word', and 'Tag' columns.
    """
    df = pd.read_excel(file_path)

    if task == "classification":
        required_columns = ["review", "sentiment"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Classification XLSX must have 'review' and 'sentiment' columns.")

        df = df.dropna(subset=required_columns)
        df['review'] = df['review'].astype(str).apply(clean_text)

        # Filter only valid sentiment labels
        valid_sentiments = {"positive", "negative"}
        df = df[df['sentiment'].isin(valid_sentiments)]

        data = [{"sentence": row["review"], "label": row["sentiment"]} for _, row in df.iterrows()]

    elif task == "ner":
        required_columns = ["Sentence #", "Word", "Tag"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError("NER XLSX must have 'Sentence #', 'Word', and 'Tag' columns.")

        df['Sentence #'] = df['Sentence #'].ffill()  # Fill missing sentence numbers
        df = df.dropna(subset=["Word", "Tag"])

        # Define valid NER tags
        valid_tags = {
            "B-art", "B-eve", "B-geo", "B-gpe", "B-nat", "B-org", "B-per", "B-tim",
            "I-art", "I-eve", "I-geo", "I-gpe", "I-nat", "I-org", "I-per", "I-tim", "O"
        }

        # Group words and tags by sentence
        grouped = df.groupby('Sentence #')
        data = []
        for _, group in grouped:
            words = group['Word'].astype(str).tolist()
            tags = group['Tag'].astype(str).tolist()
            if all(tag in valid_tags for tag in tags):
                data.append({'words': words, 'labels': tags})

    else:
        raise ValueError(f"Unsupported task type: {task}. Use 'classification' or 'ner'.")

    return data


def clean_text(text):
    """Cleans text by removing HTML tags and special characters."""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", "", text)  # Remove unwanted characters
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text


def custom_collate_fn(batch):
    """
    Custom function to batch tokenized inputs and labels.
    Ensures proper tensor stacking.
    """
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    batched_inputs = {key: torch.stack([item[key] for item in inputs]) for key in inputs[0]}
    batched_labels = torch.stack(labels)

    return batched_inputs, batched_labels
