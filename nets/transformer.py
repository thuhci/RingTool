from functools import wraps

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


def timer(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        
        result = func(self, *args, **kwargs)
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)
        
        module_name = self.__class__.__name__
        
        print(f"[Timer] Module: {module_name:<15} | GPU Time: {elapsed_time:.3f} ms")
        
        return result
    return wrapper

class MLPProjector(nn.Module):
    def __init__(self, input_dim=90, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (batch, seq_len//30, 30*3=90)
        return self.net(x)

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance=1024):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.rel_pos_bias = nn.Embedding(2*max_distance+1, num_heads)
        self.cache = {}
    
    def forward(self, q_len, k_len):
        if (q_len, k_len) not in self.cache:
            range_vec_q = torch.arange(q_len)
            range_vec_k = torch.arange(k_len)
            distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
            distance_mat_clipped = torch.clamp(distance_mat, -self.max_distance, self.max_distance)
            final_mat = distance_mat_clipped + self.max_distance
            
            bias = self.rel_pos_bias(final_mat.to(next(self.parameters()).device))
            self.cache[(q_len, k_len)] = bias.permute(2, 0, 1).unsqueeze(0).detach()
        return self.cache[(q_len, k_len)]
 
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads  # head dimension
        self.scale = self.head_dim ** -0.5  # scaling factor for dot-product attention

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
        self.rel_pos_bias = RelativePositionBias(heads)
        
        self.dropout = dropout

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        #q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [b, n, h, d]
        
        rel_bias = self.rel_pos_bias(n, n)
        
        # handle mask
        attn_mask = None
        if mask is not None:
            # original mask (including CLS token)
            mask = F.pad(mask.flatten(1), (1, 0), value=True)  # CLS token mask
            mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
            mask = mask.unsqueeze(1)  # [b, 1, n, n]
            mask = mask.expand(-1, h, -1, -1)  # [b, h, n, n]
            mask_value = torch.finfo(q.dtype).min
            mask = torch.where(mask, 0.0, mask_value)
            attn_mask = rel_bias + mask  # merge mask
        else:
            attn_mask = rel_bias.expand(b, h, n, n)  # update dim
        
        # Use Flash Attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale
        )
        
        # update out shape
        out = out.transpose(1, 2).reshape(b, n, -1)  # [b, h, n, d] -> [b, n, dim]
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class RingToolBERT(nn.Module):
    def __init__(self, in_channels=3, window_size=5,
                 dim=128, depth=4, heads=4, mlp_dim=64,
                 dropout=0.1, num_classes=1, **kw):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Unfold(kernel_size=(window_size, 1), stride=window_size),  # sliding window
            Rearrange('b (c w) l -> b l (w c)', w=window_size, c=in_channels),
            MLPProjector(input_dim=in_channels*window_size, output_dim=dim)
        )

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        # x shape: (batch, seq_len, channels)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        
        # projection [batch, new_seq_len, dim]
        x = self.projection(x)
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.transformer(x)
        
        # output [CLS] token
        cls_output = x[:, 0]
        return self.cls_head(cls_output)[:,0], x


# Helper classes
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Rearrange(nn.Module):
    def __init__(self, pattern, **kw):
        super().__init__()
        self.pattern = pattern
        self.kw = kw

    def forward(self, x):
        return rearrange(x, self.pattern, **self.kw)
