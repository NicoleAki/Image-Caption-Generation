import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pandas as pd
import pickle
from PIL import Image
import os
from models.cnn_lstm import CNN_LSTM

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
embed_size = 256
hidden_size = 512
num_layers = 1
num_epochs = 5
batch_size = 64               # safer size
learning_rate = 0.001

# ---------------- Dataset ----------------
class ImageCaptionDataset(Dataset):
    def __init__(self, df, vocab, transform=None):
        self.df = df
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.iloc[idx]['caption']
        img_path = self.df.iloc[idx]['full_path']

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Could not find image {img_path}. Skipping.")
            # Return the next available item
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            image = self.transform(image)

        # Caption → token ids
        tokens = str(caption).lower().split()

        caption_ids = [self.vocab['word2idx']['<START>']]
        caption_ids += [self.vocab['word2idx'].get(tok, self.vocab['word2idx']['<UNK>']) for tok in tokens]
        caption_ids.append(self.vocab['word2idx']['<END>'])

        return image, torch.tensor(caption_ids)

# -------------- Collate FN ----------------
def collate_fn(data):
    # Sort by caption length (descending)
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions = zip(*data)
    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded[i, :end] = cap[:end]

    return images, padded, lengths

# -------------- Training Loop ----------------
def main():

    # Load CSV and vocab
    df = pd.read_csv('data/processed/subset_10000_train.csv')
    with open('data/processed/vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageCaptionDataset(df, vocab, transform)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    # Model
    model = CNN_LSTM(embed_size, hidden_size, len(vocab['word2idx']), num_layers).to(device)

    pad_idx = vocab['word2idx']['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_steps = len(data_loader)

    model.train()  # Training mode

    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):

            images = images.to(device)
            captions = captions.to(device)

            # Prepare inputs and targets for teacher forcing
            # inputs: all tokens except the last
            inputs = captions[:, :-1]
            # targets: all tokens except the first
            targets = captions[:, 1:]

            # Forward
            outputs = model(images, inputs, lengths)

            # Flatten targets to match flattened outputs from decoder
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{total_steps}], "
                      f"Loss: {loss.item():.4f}, Perplexity: {torch.exp(loss):.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/cnn_lstm.pth')
    print("Model saved.")

if __name__ == '__main__':
    main()
