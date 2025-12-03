import torch.nn as nn
from .cnn_encoder import CNNEncoder
from .lstm_decoder import LSTMDecoder

class CNN_LSTM(nn.Module):
    """
    CNN-LSTM model.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNN_LSTM, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        """
        Forward pass.
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs

    def sample(self, images, states=None):
        """
        Generate captions for images.
        """
        features = self.encoder(images)
        sampled_ids = self.decoder.sample(features, states)
        return sampled_ids
