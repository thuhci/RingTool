import torch
from torch import nn
from einops import rearrange, repeat
from mamba_ssm import Mamba2

class MLPProjector(nn.Module):
    def __init__(self, input_dim=90, hidden_dim=128, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MambaStack(nn.Module):
    def __init__(self, dim, depth, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba2(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TaoMamba(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 window_size=5,
                 dim=128, 
                 depth=4, 
                 d_state=64,
                 d_conv=4,
                 expand=2,
                 num_classes=1, 
                 **kw):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Unfold(kernel_size=(window_size, 1)),
            Rearrange('b (c w) l -> b l (w c)', w=window_size, c=in_channels),
            MLPProjector(input_dim=in_channels*window_size, output_dim=dim)
        )
        
        self.mamba = MambaStack(dim, depth, d_state, d_conv, expand)
        
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.projection(x)  # [batch, seq_len, dim]
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat((x, cls_tokens), dim=1)  # [batch, seq_len+1, dim]
        
        x = self.mamba(x)  # [batch, seq_len+1, dim]
        
        cls_output = x[:, -1]  # [batch, dim]
        return self.cls_head(cls_output)[:,0], x

class Rearrange(nn.Module):
    def __init__(self, pattern, **kw):
        super().__init__()
        self.pattern = pattern
        self.kw = kw

    def forward(self, x):
        return rearrange(x, self.pattern, **self.kw)
