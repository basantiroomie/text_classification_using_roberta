# Roberta-based Text Classification Model

This repository provides a Roberta-based text classification model with text preprocessing, training, and evaluation. The code uses Hugging Face Transformers and datasets, Scikit-learn, NLTK for text preprocessing, and PyTorch for model training.

## Description
This project involves training a Roberta model for sequence classification on a custom dataset. The dataset is cleaned by removing duplicates, and data is preprocessed by filtering based on a specific class, removing punctuation, and stop words.

## Steps in the Pipeline
- Data Cleaning: Removing duplicates and filtering based on word count.
- Text Preprocessing: Removing punctuation and stopwords from the text.
- Model Preparation: Using Hugging Face's RobertaForSequenceClassification for training.
- Training the Model: Setting up training arguments with early stopping.
- Evaluation: Generating predictions on the test set and producing a classification report.

## Requirements

To run the code, ensure the following packages are installed:
```bash
!pip install pandas numpy nltk seaborn matplotlib scikit-learn transformers torch datasets

