# Classification---DistilBERT

Multi-class classification problem - identify the emotion label of the comment. The model will differentiate between 10 classes.

# Dataset
Dataset is not added here. 
The model was trained on a subset of GoEmotions dataset. The selected dataset was imbalanced, with 6532 instances and 10 classes. 
Due to this small imbalanced dataset we opted for DistilBERT.

# Evaluation metrics
Accuracy, macro-avergaed F1, macro-avergaed recall, macro-avergaed precision and confusion marix. 

# Methodology 

1. Data pre-process
   Load dataset. Label Encoding - convert text labels into numerical values. Tokenization is done. Convert the data into Hugging Face’s Dataset format.
2. Model set-up
   Initialize model for classification - load pre-trained model - DistilBert from Hugging Face’s transformers library. Define training arguments using 
   “TrainingArugumenst” and metrics for evaluation.
3. Training model and saving model
   Create trainer - initiate Hugging Face’s Trainer. Start trainer and evaluations. Save model based on the highest evaluation metric.
4. Testing and evaluation
   Load the saved model and weights. Do testing using the test file. Get the evaluation metrics like accuracy and macro-avergaed F1 value. 

