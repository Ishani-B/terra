import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification
import json
from tqdm import tqdm

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Custom dataset class for transcripts
class TranscriptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label

# Function to load JSON data
def load_json_data(root_dir):
    data, labels = [], []
    for category in ["ad", "healthy"]:
        label = 1 if category == "ad" else 0
        category_dir = os.path.join(root_dir, category)

        if not os.path.exists(category_dir):
            print(f"❌ Directory not found: {category_dir}")
            continue

        for subdir, _, files in os.walk(category_dir):
            for file in sorted(files):  # Sort files to ensure consistent order
                if file.endswith(".json"):
                    file_path = os.path.join(subdir, file)
                    try:
                        with open(file_path, "r") as f:
                            json_data = json.load(f)

                        transcript = " ".join(
                            seg.get("text", "").strip()
                            for seg in json_data.get("transcript", [])
                            if "text" in seg
                        )

                        if transcript:
                            data.append(transcript)
                            labels.append(label)
                    except json.JSONDecodeError:
                        print(f"❌ JSON decoding error: {file_path}")

    print(f"✅ Loaded {len(data)} transcripts.")
    return data, labels

# Load and preprocess eye-tracking data
def load_eye_data(folder, label):
    data = []
    for file in sorted(os.listdir(folder)):  # Sort files to ensure consistent order
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            data.append(df.values[0])  # Extract first row as feature vector
    data = np.array(data)
    labels = np.full((data.shape[0], 1), label)
    return np.hstack((data, labels))

# Load healthy & Alzheimer's eye-tracking data
healthy_eye_data = load_eye_data("healthy_eye", 0)
ad_eye_data = load_eye_data("ad_eye", 1)
eye_data = np.vstack((healthy_eye_data, ad_eye_data))

# Split into features & labels
X_eye = eye_data[:, :-1]
y_eye = eye_data[:, -1]

# Split dataset
X_eye_train, X_eye_test, y_eye_train, y_eye_test = train_test_split(X_eye, y_eye, test_size=0.3, random_state=42)

# Feature scaling for eye-tracking data
scaler = StandardScaler()
X_eye_train_scaled = scaler.fit_transform(X_eye_train)
X_eye_test_scaled = scaler.transform(X_eye_test)

# Train SVM for eye-tracking data
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_eye_train_scaled, y_eye_train)

# Extract eye-tracking features (decision function values)
eye_train_features = svm_model.decision_function(X_eye_train_scaled)
eye_test_features = svm_model.decision_function(X_eye_test_scaled)

# Load and preprocess speech data
root_dir = "/Users/ishanibakshi/sciencefair"  # Update this path
all_texts, all_labels = load_json_data(root_dir)

# Create dataset
dataset = TranscriptDataset(all_texts, all_labels, tokenizer)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained BERT model for classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Extract BERT features
def extract_bert_features(dataloader, model):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, _ = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            features.extend(outputs.logits.cpu().numpy())
    return np.array(features)

bert_train_features = extract_bert_features(train_loader, model)
bert_test_features = extract_bert_features(test_loader, model)

# Ensure the number of samples matches
assert len(eye_train_features) == len(bert_train_features), "Mismatch in training samples"
assert len(eye_test_features) == len(bert_test_features), "Mismatch in testing samples"

# Concatenate eye-tracking and BERT features
combined_train_features = np.hstack((eye_train_features.reshape(-1, 1), bert_train_features))
combined_test_features = np.hstack((eye_test_features.reshape(-1, 1), bert_test_features))

# Train a new classifier on combined features
combined_model = RandomForestClassifier(n_estimators=100, random_state=42)
combined_model.fit(combined_train_features, y_eye_train)

# Evaluate the combined model
y_pred_combined = combined_model.predict(combined_test_features)
print(f"Combined Model Accuracy: {accuracy_score(y_eye_test, y_pred_combined):.4f}")
print(f"Combined Model Precision: {precision_score(y_eye_test, y_pred_combined):.4f}")
print(f"Combined Model Recall: {recall_score(y_eye_test, y_pred_combined):.4f}")
print(f"Combined Model F1-Score: {f1_score(y_eye_test, y_pred_combined):.4f}")

import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification
import json
from tqdm import tqdm

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Custom dataset class for transcripts
class TranscriptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label

# Function to load JSON data
def load_json_data(root_dir):
    data, labels = [], []
    for category in ["ad", "healthy"]:
        label = 1 if category == "ad" else 0
        category_dir = os.path.join(root_dir, category)

        if not os.path.exists(category_dir):
            print(f"❌ Directory not found: {category_dir}")
            continue

        for subdir, _, files in os.walk(category_dir):
            for file in sorted(files):  # Sort files to ensure consistent order
                if file.endswith(".json"):
                    file_path = os.path.join(subdir, file)
                    try:
                        with open(file_path, "r") as f:
                            json_data = json.load(f)

                        transcript = " ".join(
                            seg.get("text", "").strip()
                            for seg in json_data.get("transcript", [])
                            if "text" in seg
                        )

                        if transcript:
                            data.append(transcript)
                            labels.append(label)
                    except json.JSONDecodeError:
                        print(f"❌ JSON decoding error: {file_path}")

    print(f"✅ Loaded {len(data)} transcripts.")
    return data, labels

# Load and preprocess eye-tracking data
def load_eye_data(folder, label):
    data = []
    for file in sorted(os.listdir(folder)):  # Sort files to ensure consistent order
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            data.append(df.values[0])  # Extract first row as feature vector
    data = np.array(data)
    labels = np.full((data.shape[0], 1), label)
    return np.hstack((data, labels))

# Load healthy & Alzheimer's eye-tracking data
healthy_eye_data = load_eye_data("healthy_eye", 0)
ad_eye_data = load_eye_data("ad_eye", 1)
eye_data = np.vstack((healthy_eye_data, ad_eye_data))

# Split into features & labels
X_eye = eye_data[:, :-1]
y_eye = eye_data[:, -1]

# Split dataset
X_eye_train, X_eye_test, y_eye_train, y_eye_test = train_test_split(X_eye, y_eye, test_size=0.3, random_state=42)

# Feature scaling for eye-tracking data
scaler = StandardScaler()
X_eye_train_scaled = scaler.fit_transform(X_eye_train)
X_eye_test_scaled = scaler.transform(X_eye_test)

# Train SVM for eye-tracking data
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_eye_train_scaled, y_eye_train)

# Extract eye-tracking features (decision function values)
eye_train_features = svm_model.decision_function(X_eye_train_scaled)
eye_test_features = svm_model.decision_function(X_eye_test_scaled)

# Load and preprocess speech data
root_dir = "/Users/ishanibakshi/sciencefair"  # Update this path
all_texts, all_labels = load_json_data(root_dir)

# Create dataset
dataset = TranscriptDataset(all_texts, all_labels, tokenizer)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained BERT model for classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Extract BERT features
def extract_bert_features(dataloader, model):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, _ = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            features.extend(outputs.logits.cpu().numpy())
    return np.array(features)

bert_train_features = extract_bert_features(train_loader, model)
bert_test_features = extract_bert_features(test_loader, model)

# Ensure the number of samples matches
assert len(eye_train_features) == len(bert_train_features), "Mismatch in training samples"
assert len(eye_test_features) == len(bert_test_features), "Mismatch in testing samples"

# Concatenate eye-tracking and BERT features
combined_train_features = np.hstack((eye_train_features.reshape(-1, 1), bert_train_features))
combined_test_features = np.hstack((eye_test_features.reshape(-1, 1), bert_test_features))

# Train a new classifier on combined features
combined_model = RandomForestClassifier(n_estimators=100, random_state=42)
combined_model.fit(combined_train_features, y_eye_train)

# Evaluate the combined model
y_pred_combined = combined_model.predict(combined_test_features)
print(f"Combined Model Accuracy: {accuracy_score(y_eye_test, y_pred_combined):.4f}")
print(f"Combined Model Precision: {precision_score(y_eye_test, y_pred_combined):.4f}")
print(f"Combined Model Recall: {recall_score(y_eye_test, y_pred_combined):.4f}")
print(f"Combined Model F1-Score: {f1_score(y_eye_test, y_pred_combined):.4f}")

print(f"Eye-tracking data shape: {eye_data.shape}")
print(f"Transcript data shape: {len(all_texts)}")

