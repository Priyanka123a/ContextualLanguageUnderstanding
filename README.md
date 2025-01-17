This project implements a Sentiment Analysis Web Application using BERT (Bidirectional Encoder Representations from Transformers) for classifying text as positive or negative. The application includes both a backend for model training and inference, as well as a web-based interface for user interaction.
Data Preprocessing:

Utilizes the IMDB movie review dataset, which contains labeled reviews as either positive or negative.
Text is tokenized and converted into features suitable for the BERT model.
Labels (positive, negative) are mapped to numerical values (1, 0).
Model Training:

The BERT model (bert-base-uncased) is fine-tuned using the dataset.
Implements a training loop with a scheduler for learning rate adjustment and an optimizer (AdamW) for gradient updates.
Supports GPU acceleration for faster training if CUDA is available.
Model Saving:

The fine-tuned BERT model and tokenizer are saved locally in the trained_bert_model directory for reuse during inference.
Web Interface:

Provides an intuitive user interface using Flask to input text for sentiment analysis.
Processes the input and returns the predicted sentiment (Positive or Negative) using the fine-tuned BERT model.
Inference API:

A REST API endpoint (/predict) accepts JSON input with a text field and returns the predicted sentiment in JSON format.
How It Works
Backend
Dataset:
The dataset (IMDB Dataset.csv) is loaded and split into training and validation sets.
Tokenization:
Text data is tokenized using the BERT tokenizer with truncation and padding.
Model:
A pre-trained BERT model is fine-tuned for binary sentiment classification.
Training:
Data is loaded in batches using PyTorch's DataLoader.
The model is trained for 3 epochs with gradient updates and learning rate scheduling.
Prediction:
User input is tokenized and passed to the model for sentiment classification.
The predicted label (0 or 1) is converted to Negative or Positive.
Frontend
A simple web page (index.html) allows users to input text for analysis.
The web page interacts with the backend through the /predict endpoint to display results
