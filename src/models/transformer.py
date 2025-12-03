import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class VisionLanguageTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, patch_size=16):
        super(VisionLanguageTransformer, self).__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.max_seq_length = max_seq_length

        # Image patch embedding
        self.patch_embedding = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encodings
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )

        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Special tokens
        self.start_token = 0 # Assuming <START> is at index 0

    def forward(self, images, captions):
        # Image processing
        img_embed = self.patch_embedding(images).flatten(2).permute(0, 2, 1)
        img_embed = self.pos_encoder(img_embed.transpose(0, 1)).transpose(0, 1)

        # Caption processing
        cap_embed = self.text_embedding(captions)
        cap_embed = self.pos_encoder(cap_embed.transpose(0, 1)).transpose(0, 1)

        # Generate masks
        tgt_mask = self.transformer.generate_square_subsequent_mask(captions.size(1)).to(images.device)

        # Transformer forward
        output = self.transformer(img_embed, cap_embed, tgt_mask=tgt_mask)
        return self.fc_out(output)

    def sample(self, images, states=None):
        batch_size = images.size(0)
        
        # Image processing
        img_embed = self.patch_embedding(images).flatten(2).permute(0, 2, 1)
        img_embed = self.pos_encoder(img_embed.transpose(0, 1)).transpose(0, 1)
        
        # Start with <START> token
        sampled_ids = torch.full((batch_size, 1), self.start_token, dtype=torch.long, device=images.device)

        for _ in range(self.max_seq_length):
            cap_embed = self.text_embedding(sampled_ids)
            cap_embed = self.pos_encoder(cap_embed.transpose(0, 1)).transpose(0, 1)

            tgt_mask = self.transformer.generate_square_subsequent_mask(sampled_ids.size(1)).to(images.device)

            output = self.transformer(img_embed, cap_embed, tgt_mask=tgt_mask)
            
            # Get the last word prediction
            last_word_logits = self.fc_out(output[:, -1, :])
            _, predicted_id = last_word_logits.max(1)
            
            # Append prediction
            sampled_ids = torch.cat([sampled_ids, predicted_id.unsqueeze(1)], dim=1)

        return sampled_ids
