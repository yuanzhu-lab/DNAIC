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
        
        # 词嵌入层
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        # TransformerDecoder层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        
        # 输出层
        self.out_layer = nn.Linear(embed_dim, vocab_size)
        
        # 初始化
        self._init_weights()
        
        # 用于自回归训练
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
        自回归解码器前向传播
        enc_output: 编码器输出 [batch_size, feat_len, embed_dim]
        seq: 输入序列 [batch_size, seq_len]
        """
        if seq is None:
            # 推理时的自回归生成
            return self._generate(enc_output)
            
        # 训练时使用teacher forcing
        # 准备目标序列，移除最后一个token用于输入，真实目标从第二个token开始
        tgt = seq[:, :-1]
        # 词嵌入和位置编码
        tgt_embed = self.word_embed(tgt) * math.sqrt(self.embed_dim)
        tgt_embed = self.pos_encoder(tgt_embed)
        
        # 准备mask
        seq_len = tgt.size(1)
        tgt_mask = self.causal_mask[:seq_len, :seq_len].to(enc_output.device)
        
        # 重排维度以匹配PyTorch Transformer的期望
        tgt_embed = tgt_embed.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        memory = enc_output.transpose(0, 1)    # [feat_len, batch_size, embed_dim]
        
        # 应用解码器
        output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
        # 重新转换回原始维度
        output = output.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        
        # 输出层
        logits = self.out_layer(output)
        
        return logits
    
    def _generate(self, enc_output):
        """推理时的自回归生成"""
        batch_size = enc_output.size(0)
        device = enc_output.device
        
        # 起始标记
        curr_ids = torch.full((batch_size, 1), 0, dtype=torch.long, device=device)
        memory = enc_output.transpose(0, 1)
        
        outputs = []
        for step in range(cfg.MODEL.SEQ_LEN-1):
            # 词嵌入
            tgt_embed = self.word_embed(curr_ids) * math.sqrt(self.embed_dim)
            tgt_embed = self.pos_encoder(tgt_embed)
            
            # 准备mask
            seq_len = curr_ids.size(1)
            tgt_mask = self.causal_mask[:seq_len, :seq_len].to(device)
            
            # 应用解码器
            tgt_embed = tgt_embed.transpose(0, 1)
            output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            output = output.transpose(0, 1)
            
            # 最后一个时间步的输出
            logits = self.out_layer(output[:, -1:])
            outputs.append(logits)
            
            # 获取预测的下一个token
            next_ids = logits.argmax(dim=-1)
            curr_ids = torch.cat([curr_ids, next_ids], dim=1)
        
        # 拼接所有输出
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