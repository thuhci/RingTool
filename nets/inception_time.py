import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    """
    Inception module with multiple parallel convolutional filters of different kernel sizes
    followed by a bottleneck layer to reduce parameters
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_bottleneck=True, use_residual=True, activation=nn.ReLU()):
        super(InceptionModule, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.use_residual = use_residual
        self.activation = activation
        self.bottleneck_channels = bottleneck_channels
        
        # Bottleneck layer
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels=in_channels, out_channels=bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False)
            in_channels_for_conv = bottleneck_channels
        else:
            self.bottleneck = None
            in_channels_for_conv = in_channels
            
        # Parallel convolutional layers with different kernel sizes
        self.conv_layers = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2  # "same" padding
            conv = nn.Conv1d(in_channels=in_channels_for_conv, out_channels=out_channels, kernel_size=k, stride=1, padding=padding, bias=False)
            self.conv_layers.append(conv)
        
        # Max pooling branch
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False)
        
        # Batch norm after concatenation
        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        
        # Residual connection
        if self.use_residual:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels * (len(kernel_sizes) + 1), kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
            )
    
    def forward(self, x):
        # Store original input for residual connection
        original_input = x
        
        # Apply bottleneck if enabled
        if self.use_bottleneck:
            x = self.bottleneck(x)
        
        # Apply parallel convolutions
        conv_outs = []
        for conv in self.conv_layers:
            conv_outs.append(conv(x))
        
        # Max pooling branch
        pool_out = self.conv_pool(self.max_pool(original_input))
        
        # Concatenate all outputs
        x = torch.cat(conv_outs + [pool_out], dim=1)
        
        # Apply batch norm and activation
        x = self.bn(x)
        x = self.activation(x)
        
        # Add residual connection if enabled
        if self.use_residual:
            x = x + self.residual(original_input)
            x = self.activation(x)
        
        return x


class InceptionBlock(nn.Module):
    """
    Block of inception modules with residual connection
    """
    def __init__(self, in_channels, out_channels, num_inception=3, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_residual=True):
        super(InceptionBlock, self).__init__()
        self.use_residual = use_residual
        
        self.inception_layers = nn.ModuleList()
        for i in range(num_inception):
            # First layer takes input_channels, others take out_channels * 4 (3 kernels + maxpool branch)
            layer_in_channels = in_channels if i == 0 else out_channels * (len(kernel_sizes) + 1)
            inception = InceptionModule(
                in_channels=layer_in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_bottleneck=True,
                use_residual=use_residual
            )
            self.inception_layers.append(inception)
    
    def forward(self, x):
        for layer in self.inception_layers:
            x = layer(x)
        return x


class InceptionTime(nn.Module):
    """
    InceptionTime model as described in:
    Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D.F., Weber, J., 
    Webb, G.I., Idoumghar, L., Muller, P.A. and Petitjean, F., 2020. 
    InceptionTime: Finding AlexNet for Time Series Classification.
    
    Input:
        X: (n_samples, n_channel, n_length) or (n_samples, n_length, n_channel)
        Y: (n_samples)
    Output:
        out: (n_samples)
    Parameters:
        in_channels: dim of input, the same as n_channel
        num_classes: number of classes for classification
        num_blocks: number of inception blocks
        kernel_sizes: kernel sizes for each parallel conv in inception module
    """
    def __init__(self, in_channels=1, out_dim=200, num_blocks=2, num_inception_per_block=3, 
                 kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_residual=True, 
                 channels_first=False, verbose=False, backbone=False):
        super(InceptionTime, self).__init__()
        self.out_dim = out_dim
        self.backbone = backbone
        self.verbose = verbose
        self.channels_first = channels_first
        
        # Feature extractor consists of multiple inception blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            # First block takes in_channels, others take output of previous block
            block_in_channels = in_channels if i == 0 else 64 * (len(kernel_sizes) + 1)
            block = InceptionBlock(
                in_channels=block_in_channels,
                out_channels=64,  # Default from the paper
                num_inception=num_inception_per_block,
                kernel_sizes=kernel_sizes,
                bottleneck_channels=bottleneck_channels,
                use_residual=use_residual
            )
            self.blocks.append(block)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        final_channels = 64 * (len(kernel_sizes) + 1)
        self.final_bn = nn.BatchNorm1d(final_channels)
        self.final_relu = nn.ReLU(inplace=True)
        
        # Classification head
        self.dense = nn.Linear(final_channels, 1)
        
        # Feature output for backbone mode
        self.dense2 = nn.Linear(final_channels, self.out_dim)
    
    def forward(self, x):
        # Handle different input formats
        if not self.channels_first:
            x = x.transpose(1, 2)  # Convert (batch, length, channels) to (batch, channels, length)
        
        if self.verbose:
            print('input shape', x.shape)
        
        # Pass through inception blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.verbose:
                print(f'after block {i+1}', x.shape)
        
        # Apply batch norm and activation
        x = self.final_bn(x)
        x = self.final_relu(x)
        
        # Global average pooling
        x = self.gap(x).squeeze(-1)
        
        if self.verbose:
            print('after GAP', x.shape)
        
        # Output based on mode
        if self.backbone:
            out = self.dense2(x)
            return None, out
        
        # Regression output
        out_value = self.dense(x)
        # Squeeze the output to (batch_size,)
        out_value = out_value.squeeze(-1)
        
        return out_value, x