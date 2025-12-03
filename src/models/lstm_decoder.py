import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMDecoder(nn.Module):
    """
    LSTM Decoder for Image Captioning.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(LSTMDecoder, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, features, captions, lengths):
        """
        features: (batch, embed_size)
        captions: (batch, seq_len)
        lengths: list of lengths for each caption
        """
        # Embed all caption words
        embeddings = self.embed(captions)  # (B, T, embed_size)

        # Prepend image features as the first input token
        features = features.unsqueeze(1)   # (B, 1, embed_size)
        embeddings = torch.cat((features, embeddings), dim=1)  # (B, T+1, embed_size)

        # Forward through LSTM over the full sequence
        hiddens, _ = self.lstm(embeddings)  # (B, T+1, H)

        # Drop the first time step (corresponding to image-only input)
        hiddens = hiddens[:, 1:, :]  # (B, T, H)

        # Linear layer on all remaining time steps
        outputs = self.linear(hiddens)  # (B, T, vocab_size)

        # Flatten batch and time dimensions to match flattened targets
        batch_size, seq_length, vocab_size = outputs.size()
        outputs = outputs.reshape(batch_size * seq_length, vocab_size)

        return outputs

    def sample(self, features, states=None):
        """
        Generate captions using greedy decoding.
        """
        sampled_ids = []
        
        # Start input = image features
        inputs = features.unsqueeze(1)  # (B, 1, embed_size)

        for _ in range(self.max_seq_length):

            # LSTM step
            hiddens, states = self.lstm(inputs, states)  # (B, 1, H)

            # Predict next word
            outputs = self.linear(hiddens.squeeze(1))   # (B, vocab_size)
            _, predicted = outputs.max(1)                # greedy choice

            sampled_ids.append(predicted)

            # Use predicted word as next input
            inputs = self.embed(predicted).unsqueeze(1)

        # Convert list to tensor
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
