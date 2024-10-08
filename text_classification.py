#!pip install pandas numpy nltk seaborn matplotlib scikit-learn transformers torch datasets

"""## Train"""

# Imports
import pandas as pd
import numpy as np
import re
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          RobertaForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
from datasets import Dataset
import torch
import nltk

nltk.download('punkt')  # For tokenization
nltk.download('stopwords')  # If you need stopwords

import pandas as pd

# Load the CSV file (adjust the file path and encoding as needed)
df = pd.read_csv('dataset/train.csv', encoding='ISO-8859-1')

# Check for duplicate rows based on 'text' and 'target' columns
# 'keep' is set to 'first' to retain the first occurrence of a duplicate and remove the rest
duplicates = df.duplicated(subset=['text', 'target'], keep='first')

# Remove the duplicate rows
df_cleaned = df[~duplicates]

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('dataset/cleaned_data.csv', index=False)

print(f"Duplicates removed based on 'text' and 'target'. Cleaned data saved to 'cleaned_data.csv'.")

df = pd.read_csv('dataset/cleaned_data.csv', encoding='ISO-8859-1')
# Step 2: Define the specific class you're interested in
specific_class = 'academic interests'  # Change this to your specific class

# Step 3: Filter the dataset for the specific class
class_df = df[df['target'] == specific_class].copy()

# Step 4: Calculate word count for each row in the 'text' column for the specific class
class_df['word_count'] = class_df['text'].apply(lambda x: len(x.split()))

# Step 5: Calculate the thresholds for top and bottom 5% for the specific class
lower_threshold = class_df['word_count'].quantile(0.15)  # Bottom 5%
upper_threshold = class_df['word_count'].quantile(0.95)  # Top 5%

# Step 6: Filter out the rows with word counts in the top and bottom 5%
filtered_class_df = class_df[(class_df['word_count'] > lower_threshold) & (class_df['word_count'] < upper_threshold)]

# Step 7: Prepare to save the filtered class with all other classes unchanged
# Filter the rest of the classes
other_classes_df = df[df['target'] != specific_class]

# Step 8: Combine the filtered specific class DataFrame with the rest of the classes
final_df = pd.concat([filtered_class_df, other_classes_df], ignore_index=True)

# Step 9: Save the resulting DataFrame to a new CSV file
final_df.to_csv('dataset/filtered.csv', index=False)

print("Filtered dataset saved as 'filtered.csv'.")

# Load dataset
df = pd.read_csv('dataset/filtered.csv', encoding='ISO-8859-1')

# Display the first few rows of the dataset
df.head()

# Display dataset information
df.info()

# Text before preprocessing
print(df['text'][1])

# Preprocessing function to remove unwanted characters
def remove_punctuations(text):
    text = re.sub(r'[\\-]', ' ', text)
    text = re.sub(r'[,.?;:\'(){}!|0-9]', '', text)
    return text

# Apply preprocessing to the text
df['text'] = df['text'].apply(remove_punctuations)

# Display the cleaned dataset
df.head()

from nltk.corpus import stopwords

# english stopwords
stopw = stopwords.words('english')
stopw[:10]

def remove_stopwords(text):
    clean_text = []
    for word in text.split(' '):
        if word not in stopw:
            clean_text.append(word)
    return ' '.join(clean_text)

# remove stopwords
df['text'] = df['text'].apply(remove_stopwords)

df.head()

torch.cuda.empty_cache()

# Split dataset into training and testing sets
train_df, test_df = train_test_split(df[['text', 'target']], train_size=0.8, shuffle=True)
test_df = test_df[:10000]  # Limit the test set to 10k for faster processing

# Display the shapes of the training and testing sets
print(train_df.shape, test_df.shape)

# Check device and set the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use Roberta tokenizer instead of BERT tokenizer
model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing and dataset pipeline functions
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=512)

def create_label_mapping(dataframe):
    unique_labels = dataframe['target'].unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    return label_mapping

def pipeline(dataframe, label_mapping):
    dataset = Dataset.from_pandas(dataframe, preserve_index=False)
    tokenized_ds = dataset.map(preprocess_function, batched=True)

    # Map the string labels to integers
    tokenized_ds = tokenized_ds.map(lambda x: {'labels': label_mapping[x['target']]})
    tokenized_ds = tokenized_ds.remove_columns('text')  # Remove text column if not needed
    return tokenized_ds

label_mapping = create_label_mapping(train_df)

# Print the label mapping
print("Label Mapping:")
for label, idx in label_mapping.items():
    print(f"{idx}: {label}")

tokenized_train = pipeline(train_df, label_mapping)
tokenized_test = pipeline(test_df, label_mapping)

# Create label mappings (label2id and id2label)
label2id = label_mapping  # label name to id
id2label = {v: k for k, v in label_mapping.items()}  # id to label name

# Use RobertaForSequenceClassification instead of BERT
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))
model.to(device)

# Add label mappings to the model's config
model.config.label2id = label2id
model.config.id2label = id2label

# Set up training arguments with early stopping
training_args = TrainingArguments(
    output_dir="./results",
    save_strategy='epoch',
    optim='adamw_torch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,  # Total number of epochs
    weight_decay=0.01,
    evaluation_strategy="epoch",  # Evaluate after each epoch
    load_best_model_at_end=True,  # Load the best model when finished training
    metric_for_best_model="eval_loss",  # Metric to compare to determine the best model
    greater_is_better=False,  # We want the eval loss to decrease
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=500,  # Log every 500 steps
    eval_steps=500,  # Evaluate every 500 steps
    save_total_limit=1,
    fp16=True,  # Enable FP16 for mixed precision training
)

# Define early stopping criteria
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Stop training if no improvement for 3 epochs
    early_stopping_threshold=0.01  # Minimum improvement threshold
)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create the Trainer with early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping_callback],  # Add the early stopping callback
)

trainer.train()  # Training step

# Prepare test dataset for predictions
tokenized_test = pipeline(test_df, label_mapping)
tokenized_test = tokenized_test.remove_columns('target')

# Get predictions
preds = trainer.predict(tokenized_test)

# Create a reverse mapping for classification report
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Convert the test labels to numeric
test_df['numeric_labels'] = test_df['target'].map(label_mapping)

# Process predictions for classification report
preds_flat = [np.argmax(x) for x in preds[0]]

# Generate the classification report
print(classification_report(test_df['numeric_labels'], preds_flat))

# Save the model
trainer.save_model('modelsroberta')


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch
from torch.cuda.amp import autocast
from transformers import AutoConfig

# Specify the path to your model directory
model_path = 'modelsroberta'

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model configuration to get class labels
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Extract class labels from config
class_labels = config.id2label  # This should map class indices to label names

print("Class labels from config:", class_labels)
print("Model and tokenizer loaded successfully.")

print("Loading test dataset...")
test_df = pd.read_csv('dataset/test.csv', encoding='ISO-8859-1')
print(test_df.head())  # Display the first few rows of the DataFrame

# Ensure 'text' column exists; modify if your text column has a different name
test_texts = test_df['text'].tolist()
test_index = test_df['Index'].tolist()

# Set device to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict in batches
def predict_in_batches(texts, batch_size=32):
    total_samples = len(texts)
    number_of_batches = (total_samples + batch_size - 1) // batch_size  # Calculate the number of batches
    print(f"Total samples: {total_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {number_of_batches}")

    predicted_classes = []
    for i in range(0, len(texts), batch_size):
        print(f"Processing batch {(i // batch_size + 1)} of {number_of_batches}")
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            with autocast():  # Enable mixed precision
                outputs = model(**inputs)

        predicted_classes.extend(torch.argmax(outputs.logits, axis=1).cpu().numpy())

    return predicted_classes


# Perform inference
predicted_classes = predict_in_batches(test_texts)

# Map predicted class indices to class labels using the config
predicted_labels = [class_labels[idx] for idx in predicted_classes]

# Print or save the predicted labels
#print(predicted_labels)

# Create a DataFrame with predictions
output_df = pd.DataFrame({
    'target': [class_labels[idx] for idx in predicted_classes],
    'Index': test_index[:len(predicted_classes)],
    #'text': test_texts[:len(predicted_classes)]
})

# Save predictions to a CSV file
output_df.to_csv('predictionsrob.csv', index=False)

# Display output
print(output_df)

