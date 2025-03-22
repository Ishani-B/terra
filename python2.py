import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class TranscriptDataset(Dataset):
    """Custom dataset class for transcripts."""
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

def load_json_data(root_dir):
    """Recursively loads JSON data from 'ad' and 'healthy' directories."""
    data, labels = [], []
    for category in ["ad", "healthy"]:
        label = 1 if category == "ad" else 0
        category_dir = os.path.join(root_dir, category)

        if not os.path.exists(category_dir):
            print(f"❌ Directory not found: {category_dir}")
            continue

        for subdir, _, files in os.walk(category_dir):
            for file in files:
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

# Load and tokenize data
root_dir = "/Users/ishanibakshi/sciencefair"
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

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Learning rate scheduler
num_epochs = 5  # Maximum number of epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_loader)
)

# Training loop with early stopping based on F1 score
best_f1 = 0
train_losses, val_losses = [], []
f1_scores = []
accuracies = []
precisions = []
recalls = []
auc_scores = []
early_stop_threshold = 0.9  # Early stopping threshold for F1 score
min_epochs = 5  # Minimum number of epochs to train

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_losses.append(val_loss / len(test_loader))
    
    # Calculate metrics
    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_preds)

    f1_scores.append(f1)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    auc_scores.append(auc_score)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, AUC: {auc_score:.4f}")

    # Save best model
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_bert_model.pth")

    # Early stopping if F1 score reaches the threshold after minimum epochs
    if f1 >= early_stop_threshold and epoch >= min_epochs:
        print(f"\nReached {early_stop_threshold} F1 score after {min_epochs} epochs, stopping training!")
        break

print("Training complete.")

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.show()

# Plot Accuracy, Precision, Recall, and AUC over epochs
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(accuracies, label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(precisions, label='Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Precision Over Epochs')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(recalls, label='Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.title('Recall Over Epochs')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(auc_scores, label='AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.title('AUC Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# Plot F1 score over epochs
plt.figure(figsize=(10, 6))
plt.plot(f1_scores, label='F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Score Over Epochs')
plt.legend()
plt.show()

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(all_labels, all_preds)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Attention Weights (Example for one sample)
def get_attention_weights(model, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), output_attentions=True)
        attention_weights = outputs.attentions[-1].squeeze(0).mean(dim=0).cpu().numpy()
    return attention_weights

sample_input_ids, sample_attention_mask, _ = test_dataset[0]
attention_weights = get_attention_weights(model, sample_input_ids.to(device), sample_attention_mask.to(device))

plt.figure(figsize=(10, 6))
sns.heatmap(attention_weights, cmap='viridis', annot=True, fmt='.2f')
plt.xlabel('Time Steps')
plt.ylabel('Attention Heads')
plt.title('Attention Weights')
plt.show()

# Gradient-Based Feature Importance
def compute_gradients(input_data, model):
    input_data = input_data.to(device).float().requires_grad_()  # Convert to float and require gradients
    model.zero_grad()
    outputs = model(input_data.unsqueeze(0))
    loss = criterion(outputs.logits, torch.argmax(outputs.logits, dim=1))
    loss.backward()
    gradients = input_data.grad.cpu().numpy()
    return gradients

gradients = compute_gradients(sample_input_ids, model)
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(gradients).mean(axis=0), cmap='viridis', aspect='auto')
plt.xlabel('Time Steps')
plt.ylabel('Tokens')
plt.colorbar(label='Gradient Magnitude')
plt.title('Gradient-Based Feature Importance')
plt.show()

