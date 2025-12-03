import os
import pickle

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from train import ImageCaptionDataset, collate_fn
from models.cnn_lstm import CNN_LSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_glove_embeddings(glove_path, vocab, embedding_dim):
    """Build embedding matrix aligned with vocab['word2idx'] using GloVe.

    Returns (embedding_matrix, matched, vocab_size).
    """
    print(f"Loading GloVe embeddings from {glove_path} ...")
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.rstrip().split(" ")
            if len(values) <= embedding_dim:
                continue
            word = values[0]
            try:
                coefs = torch.tensor([float(v) for v in values[1:1+embedding_dim]], dtype=torch.float)
            except ValueError:
                continue
            if coefs.numel() != embedding_dim:
                continue
            embeddings_index[word] = coefs

    word2idx = vocab["word2idx"]
    vocab_size = len(word2idx)
    embedding_matrix = torch.randn(vocab_size, embedding_dim) * 0.01

    matched = 0
    for word, idx in word2idx.items():
        vec = embeddings_index.get(word.lower())
        if vec is not None:
            embedding_matrix[idx] = vec
            matched += 1

    # Set PAD token to zeros if present
    if "<PAD>" in word2idx:
        embedding_matrix[word2idx["<PAD>"]] = torch.zeros(embedding_dim)

    print(f"GloVe coverage: {matched}/{vocab_size} words ({matched / vocab_size * 100:.2f}%)")
    return embedding_matrix, matched, vocab_size


def main():
    # Hyperparameters for GloVe-based model
    embed_size = 100  # GloVe 6B 100d
    hidden_size = 512
    num_layers = 1
    num_epochs = 5
    batch_size = 64
    learning_rate = 1e-3

    glove_path = "data/embeddings/glove.6B.100d.txt"

    # Load CSV and vocab (use full_dataset split for more data)
    df = pd.read_csv("data/processed/full_dataset_train.csv")
    with open("data/processed/vocabulary.pkl", "rb") as f:
        vocab = pickle.load(f)

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
        collate_fn=collate_fn,
    )

    # Build embedding matrix from GloVe
    embedding_matrix, matched, vocab_size = load_glove_embeddings(glove_path, vocab, embed_size)

    # Model
    model = CNN_LSTM(embed_size, hidden_size, vocab_size, num_layers).to(device)

    # Initialize decoder embeddings with GloVe and freeze them (non-trainable)
    model.decoder.embed.weight.data.copy_(embedding_matrix)
    model.decoder.embed.weight.requires_grad = False

    pad_idx = vocab["word2idx"]["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Only optimize parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    total_steps = len(data_loader)
    model.train()

    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)

            # inputs: all but last token, targets: all but first
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(images, inputs, lengths)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{total_steps}], "
                    f"Loss: {loss.item():.4f}, Perplexity: {torch.exp(loss):.4f}"
                )

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_lstm_glove.pth")
    print("GloVe-based CNN+LSTM model saved to models/cnn_lstm_glove.pth")


if __name__ == "__main__":
    main()
