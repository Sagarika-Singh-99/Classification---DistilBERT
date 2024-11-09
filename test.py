# load libraries

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

#Model paths and test file path

base_dir = '/home/ss/'

output_dir = os.path.join(base_dir, 'best_model_ps2')

#test file path add here- 
dataset_path = os.path.join(base_dir, 'T1_test.csv')

model_save_path = os.path.join(base_dir, 'saved_model')
torch_weights_path = os.path.join(base_dir, 'model_weights.pth')

#Output test file path:
output_path = os.path.join(base_dir, 'test_output_with_predictions.csv')

# Load the saved model and tokenizer from the directory
model_save_path = '/home/ss/saved_model'  

model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_save_path)

print("Model and tokenizer loaded successfully.")

# Load the new test dataset
df_test = pd.read_csv(dataset_path)

# Encode the labels to numerical values (ensure same encoding as training)
label_encoder = LabelEncoder()
df_test['label'] = label_encoder.fit_transform(df_test['label'])

# Tokenize the test text data
test_encodings = tokenizer(df_test['text'].tolist(), truncation=True, padding=True, max_length=512)

# Convert the test data into Hugging Face Dataset format
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': df_test['label'].tolist()
})


# Define metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    acc = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
    macro_f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro')
    return {'accuracy': acc, 'macro_f1': macro_f1}

# Increase evaluation batch size and shorten max token length
training_args = TrainingArguments(
    per_device_eval_batch_size=64,  # Increase batch size
    output_dir=model_save_path,
    logging_dir=os.path.join(base_dir, 'logs'),
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Evaluate on a smaller test dataset for faster checks
small_test_dataset = test_dataset.shuffle(seed=42).select(range(100))
eval_results = trainer.evaluate(eval_dataset=small_test_dataset)

# Print evaluation metrics
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Macro-averaged F1 Score: {eval_results['eval_macro_f1']:.4f}")

from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Calculate and display confusion matrix, macro-averaged precision, and recall
conf_matrix = confusion_matrix(true_labels, pred_labels)
macro_precision = precision_score(true_labels, pred_labels, average='macro')
macro_recall = recall_score(true_labels, pred_labels, average='macro')

print("Confusion Matrix:")
print(conf_matrix)
print(f"Macro-averaged Precision: {macro_precision:.4f}")
print(f"Macro-averaged Recall: {macro_recall:.4f}")



