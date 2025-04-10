import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lib.config import cfg

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, depth=3, num_heads=8, dropout=0.1, ff_dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Word embedding layer
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        # TransformerDecoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        
        # Output layer
        self.out_layer = nn.Linear(embed_dim, vocab_size)
        
        # Initialization
        self._init_weights()
        
        # For autoregressive training
        self.register_buffer('causal_mask', self._generate_square_subsequent_mask(cfg.MODEL.SEQ_LEN))
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, enc_output, seq=None):
        """
        Autoregressive decoder forward pass
        enc_output: encoder output [batch_size, feat_len, embed_dim]
        seq: input sequence [batch_size, seq_len]
        """
        if seq is None:
            # Autoregressive generation during inference
            return self._generate(enc_output)
            
        # Teacher forcing during training
        # Prepare target sequence, remove last token for input, real target starts from second token
        tgt = seq[:, :-1]
        # Word embedding and positional encoding
        tgt_embed = self.word_embed(tgt) * math.sqrt(self.embed_dim)
        tgt_embed = self.pos_encoder(tgt_embed)
        
        # Prepare mask
        seq_len = tgt.size(1)
        tgt_mask = self.causal_mask[:seq_len, :seq_len].to(enc_output.device)
        
        # Reorder dimensions to match PyTorch Transformer expectations
        tgt_embed = tgt_embed.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        memory = enc_output.transpose(0, 1)    # [feat_len, batch_size, embed_dim]
        
        # Apply decoder
        output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
        # Convert back to original dimensions
        output = output.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        
        # Output layer
        logits = self.out_layer(output)
        
        return logits
    
    def _generate(self, enc_output):
        """Autoregressive generation during inference"""
        batch_size = enc_output.size(0)
        device = enc_output.device
        
        # Start token
        curr_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
        memory = enc_output.transpose(0, 1)
        
        outputs = []
        for step in range(cfg.MODEL.SEQ_LEN-1):
            # Word embedding
            tgt_embed = self.word_embed(curr_ids) * math.sqrt(self.embed_dim)
            tgt_embed = self.pos_encoder(tgt_embed)
            
            # Prepare mask
            seq_len = curr_ids.size(1)
            tgt_mask = self.causal_mask[:seq_len, :seq_len].to(device)
            
            # Apply decoder
            tgt_embed = tgt_embed.transpose(0, 1)
            output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            output = output.transpose(0, 1)
            
            # Output of the last time step
            logits = self.out_layer(output[:, -1:])
            outputs.append(logits)
            
            # Get predicted next token
            next_ids = logits.argmax(dim=-1)
            curr_ids = torch.cat([curr_ids, next_ids], dim=1)
        
        # Concatenate all outputs
        logits = torch.cat(outputs, dim=1)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 