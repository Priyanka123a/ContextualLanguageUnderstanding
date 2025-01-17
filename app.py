import pandas as pd
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

app = Flask(__name__)

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the dataset
DATA_PATH = "IMDB Dataset.csv"  # Replace with your dataset path
df = pd.read_csv(DATA_PATH)

# Dataset Preprocessing
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['review'].tolist(), df['sentiment'].tolist(), test_size=0.2, random_state=42
)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encodings = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        return {key: val.squeeze(0) for key, val in encodings.items()}, torch.tensor(label)

train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Training Loop
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Train
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Save the trained model
model.save_pretrained("trained_bert_model")
tokenizer.save_pretrained("trained_bert_model")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    text = input_data.get('text', '')

    # Preprocess and predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).item()
    sentiment = "Positive" if predictions == 1 else "Negative"
    return jsonify({'sentiment': sentiment})


if __name__ == "__main__":
    app.run(debug=True)
