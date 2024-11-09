# install libraries

pip install transformers datasets torch pandas sklearn

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from torch.utils.data import DataLoader
import os

os.environ["WANDB_DISABLED"] = "true"

# Define paths for saving models and dataset location 

base_dir = '/home/ss/'

output_dir = os.path.join(base_dir, 'best_model_ps2')
dataset_path = os.path.join(base_dir, 'T1_train.csv')
model_save_path = os.path.join(base_dir, 'saved_model')
torch_weights_path = os.path.join(base_dir, 'model_weights.pth')

# Load the dataset
df = pd.read_csv(dataset_path)

# Encode the labels to numerical values
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split the dataset into training and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the text data
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512)

# Convert into Hugging Face Dataset format
train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_labels.tolist()})
test_dataset = Dataset.from_dict({'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask'], 'labels': test_labels.tolist()})

# Load the pre-trained DistilBERT model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(df['label'].unique()))

# Define metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    acc = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
    macro_f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
    return {'accuracy': acc, 'macro_f1': macro_f1}

# Set training arguments
training_args = TrainingArguments(
    output_dir=output_dir,  
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,  # Keep only the best model
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=os.path.join(base_dir, 'logs'),  
    logging_steps=10,
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_macro_f1",  # Choose the best model based on macro F1 score
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model and save the best model to Google Drive
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print evaluation metrics
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Macro-averaged F1 Score: {eval_results['eval_macro_f1']:.4f}")

# Save the model's state_dict using torch
torch.save(model.state_dict(), torch_weights_path)
print("Model weights saved successfully using torch.")

# Initialize a new model with the same architecture
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=10)

# Load the model's state_dict using torch
model.load_state_dict(torch.load(torch_weights_path))
print("Model loaded successfully using torch.")

# Save the model and tokenizer using Hugging Face's save_pretrained method
model.save_pretrained(model_save_path)  # Save the model's weights and configuration
tokenizer.save_pretrained(model_save_path)  # Save the tokenizer
print("Model and tokenizer saved successfully.")

# Load the saved model
model = DistilBertForSequenceClassification.from_pretrained(model_save_path)

# Load the saved tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_save_path)
print("Model and tokenizer loaded successfully.")
