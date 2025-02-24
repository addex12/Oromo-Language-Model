# Afaan Oromo-Language-Model

# Afaan Oromo Translation Model

## Overview
This repository provides a **machine translation model** that translates between **English** and **Afaan Oromo**. The model is based on the Hugging Face **transformers** library and uses pre-trained models that are fine-tuned on English ↔ Afaan Oromo datasets.

### Key Features:
- **Translation** between English and Afaan Oromo
- Fine-tuning of pre-trained models for improved accuracy
- Preprocessing and tokenization for datasets
- Easy-to-use interface for translation tasks

---

## Installation

To get started with this project, follow the steps below:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/afaan-oromo-translation.git
   cd afaan-oromo-translation

    Create and activate a Python virtual environment:

python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate  # For Windows

Install the required libraries:

    pip install -r requirements.txt

How to Use
Translating Text:

You can use the pre-trained model for quick translation:

from transformers import MarianMTModel, MarianTokenizer

# Load the pre-trained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-orm"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Translate text
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
translated = model.generate(**inputs)
output = tokenizer.decode(translated[0], skip_special_tokens=True)

print(output)  # Output: Afaan Oromo translation

Fine-tuning the Model:

If you want to fine-tune the model with your custom dataset, follow these steps:

    Prepare your English ↔ Afaan Oromo dataset in CSV format (with columns english_text and afaan_oromo_text).

    Load and preprocess the dataset:

from datasets import load_dataset

dataset = load_dataset("csv", data_files={"train": "path/to/your_dataset.csv"})
train_data = dataset["train"]

Train the model:

    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir="./afaan_oromo_translator",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
    )

    trainer.train()

Contributing

Contributions are welcome! If you'd like to contribute to the project, feel free to fork the repository and submit a pull request.
License

This project is licensed under the MIT License.
Contact

If you have any questions, feel free to contact me:

    Name: Adugna Gizaw
    Email: adugna.gizaw@flipperschools.com

Acknowledgements

    Hugging Face: for providing pre-trained models and the Transformers library
    TensorFlow & PyTorch: for the machine learning frameworks
