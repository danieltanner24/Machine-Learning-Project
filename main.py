import json
import os
import pandas as pd
import gzip
from json.decoder import JSONDecodeError
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch

def load_jsonl_files(directory_path):
    data = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as file:
            for line in file:
                try:
                    data.append(json.loads(line))
                except JSONDecodeError:
                    print(f"Error decoding JSON in file {file_name}: {line}")
                    continue
    return data

# path to the extracted dataset
dataset_path = 'C:\\Users\\Daniel\\Downloads\\python\\final\\jsonl\\train'

# Load the dataset
code_search_net_data = load_jsonl_files(dataset_path)

# Convert the dataset to a pandas DataFrame for easier processing
code_search_net_df = pd.DataFrame(code_search_net_data)

print(code_search_net_data[0].keys())

# Assuming your dataset is a list of dictionaries called 'code_search_net_data'
data = np.array(code_search_net_data)

# Split the data into training and non-training sets (validation + testing)
train_data, non_train_data = train_test_split(data, test_size=0.2, random_state=42)

# Split the non-training data into validation and testing sets
val_data, test_data = train_test_split(non_train_data, test_size=0.5, random_state=42)

# Convert the data back to lists of dictionaries
train_data = train_data.tolist()
val_data = val_data.tolist()
test_data = test_data.tolist()

print(f"Train set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")
print(f"Test set size: {len(test_data)}")

# Initialize the GPT-2 tokenizer and modelF
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to be the same as the EOS token
config = GPT2Config.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

# Define a function to tokenize and encode code examples
def encode_code_examples(tokenizer, code_examples):
    return tokenizer.batch_encode_plus(
        code_examples,
        padding=True,
        max_length=1024,  # Set the maximum length for the sequences.
        return_tensors="pt",
        truncation=True,  # Truncate the sequences if they exceed the maximum length.
    )

def custom_collate(batch):
    max_length = max([item["input_ids"].shape[0] for item in batch])

    input_ids = []
    attention_mask = []

    for item in batch:
        pad_length = max_length - item["input_ids"].shape[0]
        input_ids.append(F.pad(item["input_ids"], (0, pad_length), 'constant', 0))
        attention_mask.append(F.pad(item["attention_mask"], (0, pad_length), 'constant', 0))

    input_ids = torch.stack(input_ids, dim=0)
    attention_mask = torch.stack(attention_mask, dim=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


# Define a PyTorch Dataset class for the code examples
class CodeExamplesDataset(Dataset):
    def __init__(self, tokenizer, code_examples):
        self.tokenizer = tokenizer
        self.code_examples = code_examples

    def __len__(self):
        return len(self.code_examples)

    def __getitem__(self, idx):
        encoded_code = encode_code_examples(self.tokenizer, [self.code_examples[idx]])
        input_ids = torch.tensor(encoded_code['input_ids'][0], dtype=torch.long)
        attention_mask = torch.tensor(encoded_code['attention_mask'][0], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# Extract code examples from the train, val, and test sets
train_code_examples = [item["code"] for item in train_data]
val_code_examples = [item["code"] for item in val_data]
test_code_examples = [item["code"] for item in test_data]

# Create PyTorch datasets and dataloaders
train_dataset = CodeExamplesDataset(tokenizer, train_code_examples)
val_dataset = CodeExamplesDataset(tokenizer, val_code_examples)
test_dataset = CodeExamplesDataset(tokenizer, test_code_examples)

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)





# Now, you can fine-tune the GPT-2 model on the training set
# You can use PyTorch's built-in training loop or any other library of your choice, such as Hugging Face's Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
print(torch.version.cuda)
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# Define the loss function, optimizer, and learning rate scheduler
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

# Training loop
def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs = batch["input_ids"].to(device)
        labels = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

# Validation loop
def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            labels = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Testing loop
def test(model, dataloader, device):
    model.eval()
    generated_code = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(inputs, attention_mask=attention_mask)
            generated_code.extend(outputs)

    return generated_code

# Train the model for a few epochs
n_epochs = 1 # was 5
for epoch in range(n_epochs):
    train_loss = train(model, train_dataloader, loss_fn, optimizer, device)
    val_loss = validate(model, val_dataloader, loss_fn, device)
    print(f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
    scheduler.step()

# Test the model
generated_code = test(model, test_dataloader, device)

model.save_pretrained("codeGen_1.0")
tokenizer.save_pretrained("codeGen_1.0")
