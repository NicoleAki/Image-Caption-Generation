import torch
import torch.nn as nn
import pandas as pd
import pickle
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from models.transformer import VisionLanguageTransformer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Dataset and Dataloader ---
class ImageCaptionDataset(Dataset):
    def __init__(self, df, vocab, transform=None):
        self.df = df
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['full_path']
        caption = self.df.iloc[idx]['caption']

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Could not find image {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        tokens = [self.vocab['word2idx'].get(word, self.vocab['word2idx']['<UNK>']) for word in caption.split()]        
        caption_tensor = torch.tensor(tokens)
        return image, caption_tensor

def collate_fn(data):
    images, captions = zip(*data)
    images = torch.stack(images, 0)
    
    lengths = [len(cap) for cap in captions]
    padded_captions = pad_sequence(captions, batch_first=True, padding_value=0) # Pad with 0 (<PAD> token)

    return images, padded_captions, lengths

# -------------- Training Loop ----------------
def main():
    # Hyperparameters
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    max_seq_length = 50
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.0001

    # Load data
    df = pd.read_csv('data/processed/subset_10000_clean_train.csv')
    with open('data/processed/vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab['word2idx'])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageCaptionDataset(df, vocab, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Model
    model = VisionLanguageTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=max_seq_length
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore <PAD> token
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            outputs = model(images, captions[:, :-1]) # Exclude <END> token for input
            targets = captions[:, 1:] # Exclude <START> token for target
            
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/transformer.pth')
    print('Transformer model saved.')

if __name__ == '__main__':
    main()
