import os
import pandas as pd
from datasets import load_dataset

# Define paths
RAW_DATA_PATH = "data/raw/afaan_oromo_dataset.csv"

# Ensure the directory exists
os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

# Load dataset from Hugging Face
try:
    print("Downloading dataset...")
    dataset = load_dataset("mozilla-foundation/common_voice_11_0", "or")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    exit(1)

# Extract text data
print("Processing dataset...")
try:
    # Convert the dataset to a list of strings
    texts = [str(text) for text in dataset['train']['sentence']]  # Ensure it's a standard list of strings
except KeyError as e:
    print(f"Error processing dataset: {e}")
    exit(1)

# Create DataFrame with translations (for now, target_text is empty)
df = pd.DataFrame({'source_text': texts, 'target_text': [''] * len(texts)})

# Save to CSV
try:
    df.to_csv(RAW_DATA_PATH, index=False, encoding='utf-8')
    print(f"Dataset saved to {RAW_DATA_PATH}")
except Exception as e:
    print(f"Error saving dataset: {e}")