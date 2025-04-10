import torch
import torch.nn as nn
from lib.config import cfg
from lib.utils import get_clip_mat

class LightPASM(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super(LightPASM, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.half_seq = seq_len // 2
        
        self.first_half_weights = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, embed_dim)) 
            for _ in range(self.half_seq)
        ])
        self.first_half_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(embed_dim)) 
            for _ in range(self.half_seq)
        ])
        
        self.second_half_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        if seq_len != self.seq_len:
            return x
        
        first_half_outputs = []
        for i in range(self.half_seq):
            pos_input = x[:, i, :]
            pos_output = torch.matmul(pos_input, self.first_half_weights[i]) + self.first_half_bias[i]
            first_half_outputs.append(pos_output.unsqueeze(1))
        
        first_half_tensor = torch.cat(first_half_outputs, dim=1)
        
        second_half_outputs = []
        for i in range(self.half_seq, self.seq_len):
            relative_idx = i - self.half_seq
            if relative_idx < self.half_seq:
                src_tensor = first_half_tensor[:, relative_idx, :]
                pos_output = self.second_half_linear(src_tensor)
                second_half_outputs.append(pos_output.unsqueeze(1))
            else:
                src_tensor = first_half_tensor[:, -1, :]
                pos_output = self.second_half_linear(src_tensor)
                second_half_outputs.append(pos_output.unsqueeze(1))
        
        second_half_tensor = torch.cat(second_half_outputs, dim=1)
        
        output = torch.cat([first_half_tensor, second_half_tensor], dim=1)
        return output

class Decoder_NA(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_dim=512, 
        depth=3,
        num_heads=8,
        dropout=0.1, 
        ff_dropout=0.1
    ):
        super(Decoder_NA, self).__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.layers = nn.ModuleList([])
        self.embed_dim = embed_dim
        for i in range(depth):
            sublayer = DecoderLayer( 
                embed_dim = embed_dim, 
                num_heads = num_heads, 
                dropout = dropout, 
                ff_dropout = ff_dropout
            )
            self.layers.append(sublayer)
        
        self.word_embed = nn.Embedding(self.vocab_size, self.embed_dim)   
        self.generator = nn.Linear(self.embed_dim, self.vocab_size, bias=True)
        self.img2text = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.1),
        )
        self.softmax = nn.Softmax(-1)

        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        self.w = torch.from_numpy(get_clip_mat(17, 144)).to(torch.float32).to(device)
        self.a_embed = nn.Linear(self.embed_dim+self.embed_dim, self.embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(self.embed_dim)
        self.light_pasm = LightPASM(embed_dim=self.embed_dim, seq_len=cfg.MODEL.SEQ_LEN)

    def forward(self, encoder_out):   
        batch_size = encoder_out.shape[0]
        seq_len = cfg.MODEL.SEQ_LEN
        text_emb = self.img2text(encoder_out)
        vocab = self.word_embed.weight
        word_prob = text_emb @ vocab.t()
        word_prob = self.softmax(word_prob)
        word_emb = text_emb + word_prob @ vocab
        w = self.w.to(word_emb.dtype)
        word_emb = torch.matmul(w.unsqueeze(0), word_emb)
        
        x = self.light_pasm(word_emb)
        
        for layer in self.layers:
            x, x_self = layer(x, encoder_out)

        x_cat = torch.cat([x, x_self], 2)
        g = self.sigmoid(self.a_embed(x_cat))
        x = self.layernorm(x*g + x_self*(1-g))
        out = self.generator(x)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, ff_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.word_attn = MultiHeadSelfAttention(
            embed_dim = embed_dim, 
            num_heads = num_heads
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.cross_att = MultiHeadSelfAttention(
            embed_dim = embed_dim, 
            num_heads = num_heads
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ff_layer = FeedForward(
            embed_dim = embed_dim, 
            ffn_embed_dim = embed_dim * 4, 
            relu_dropout = ff_dropout
        )
        self.layer_norm3 = torch.nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out):
        short_cut = x
        x = self.word_attn(
            q = x,
            k = x,
            v = x
        )
        x_self = x
        x = self.dropout(x)
        x = self.layer_norm1(x + short_cut)

        short_cut = x            
        x = self.cross_att(
            q = x,
            k = encoder_out,
            v = encoder_out
        )
        x = self.dropout(x)
        x = self.layer_norm2(x + short_cut)
        
        short_cut = x
        x = self.ff_layer(x)
        x = self.dropout(x)
        x = self.layer_norm3(x + short_cut)
        
        return x, x_self
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.o_linear = nn.Linear(embed_dim, embed_dim)
        
        self.softmax = nn.Softmax(-1)
    
    def forward(self, q, k, v, mask=None):
        B_, N, C = q.size()
        q = self.q_linear(q).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.o_linear(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = nn.Dropout(relu_dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x