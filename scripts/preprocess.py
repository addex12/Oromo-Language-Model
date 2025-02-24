import os
import pandas as pd
import re
from transformers import AutoTokenizer

# Define file paths
RAW_DATA_PATH = "data/raw/your_dataset.csv"  # Update with your dataset file
PROCESSED_DATA_PATH = "data/processed/processed_dataset.csv"
TOKENIZER_NAME = "facebook/mbart-large-50"  # Supports multiple languages, including Afaan Oromo

# Load dataset
def load_dataset(file_path):
    """Loads dataset from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)

# Text cleaning function
def clean_text(text):
    """Removes special characters, extra spaces, and converts to lowercase."""
    text = text.lower()
    text = re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿሀ-ፐ ]", "", text)  # Keeps Afaan Oromo and Latin characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Tokenization function
def tokenize_data(df, tokenizer):
    """Tokenizes input and target texts."""
    df["source_tokens"] = df["source_text"].apply(lambda x: tokenizer.encode(x, truncation=True, padding="max_length", max_length=128))
    df["target_tokens"] = df["target_text"].apply(lambda x: tokenizer.encode(x, truncation=True, padding="max_length", max_length=128))
    return df

def preprocess_and_save():
    """Loads, cleans, tokenizes, and saves the dataset."""
    print("Loading dataset...")
    df = load_dataset(RAW_DATA_PATH)
    
    print("Cleaning text...")
    df["source_text"] = df["source_text"].apply(clean_text)
    df["target_text"] = df["target_text"].apply(clean_text)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    print("Tokenizing data...")
    df = tokenize_data(df, tokenizer)
    
    print("Saving processed dataset...")
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(f"✅ Preprocessing complete! Processed dataset saved at {PROCESSED_DATA_PATH}")

# Run preprocessing
if __name__ == "__main__":
    preprocess_and_save()
