import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups)

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.max_pool(net)
        return net


class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    # Modify __init__ to accept dropout_p
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, dropout_p=0.5, is_first_block=False):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.dropout_p = dropout_p

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        # Use the dropout_p parameter here
        self.do1 = nn.Dropout(p=self.dropout_p)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        # Use the dropout_p parameter here
        self.do2 = nn.Dropout(p=self.dropout_p)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        # shortcut
        out += identity
        return out


class ResNet1D(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length) -> Changed to (n_samples, n_length, n_channel) based on forward
        Y: (n_samples)
    Output:
        out: (n_samples) for regression or (n_samples, n_classes) for classification
    Pararmetes:
        in_channels: dim of input features (e.g., 5 for ir, red, ax, ay, az)
        base_filters: number of filters in the first several Conv layer, it will double at every increasefilter_gap blocks
        kernel_size: width of kernel
        stride: stride of kernel moving (in BasicBlock)
        groups: set largely to 1 for ResNet, >1 for ResNeXt
        n_block: number of BasicBlocks
        downsample_gap: how many blocks between downsampling layers
        increasefilter_gap: how many blocks between increasing filter count
        use_bn: whether to use BatchNorm
        use_do: whether to use Dropout in blocks
        dropout_p: the probability for dropout layers in blocks (NEW)
        verbose: print shapes during forward pass
        backbone: if True, return features before final dense layer
        output_dim: output dimension (1 for regression, n_classes for classification) - Modified default to 1 for regression
    """
    # Add dropout_p to __init__
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, dropout_p=0.5, use_final_do=False, final_dropout_p=0.5, verbose=False, backbone=False, output_dim=1): # Default output_dim=1
        super(ResNet1D, self).__init__()
        self.out_dim = output_dim
        self.backbone = backbone
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.dropout_p = dropout_p
        self.use_final_do = use_final_do
        self.final_dropout_p = final_dropout_p
        self.downsample_gap = downsample_gap
        self.increasefilter_gap = increasefilter_gap

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            is_first_block = i_block == 0
            downsample = (i_block + 1) % self.downsample_gap == 0 # Corrected logic? Check if this matches original intent
            
            # Correctly calculate in_channels and out_channels based on previous block's output
            current_in_channels = base_filters * (2**((i_block) // self.increasefilter_gap))
            current_out_channels = base_filters * (2**((i_block + 1) // self.increasefilter_gap)) # Simpler way to think about it

            # Ensure correct channel handling for the very first block
            if is_first_block:
                 current_in_channels = base_filters
                 current_out_channels = base_filters # Usually doesn't change channels in the first block unless increasefilter_gap=1

            # Refined channel calculation logic
            block_in_channels = out_channels # Channels from the previous layer/block
            # Increase filters every increasefilter_gap blocks
            if (i_block > 0) and (i_block % self.increasefilter_gap == 0):
                 block_out_channels = block_in_channels * 2
            else:
                 block_out_channels = block_in_channels

            tmp_block = BasicBlock(
                in_channels=block_in_channels, # Use channels from previous layer
                out_channels=block_out_channels, # Calculated output channels
                kernel_size=self.kernel_size,
                stride = self.stride, # Pass stride defined for ResNet1D
                groups = self.groups,
                downsample=downsample,
                use_bn = self.use_bn,
                use_do = self.use_do,
                dropout_p = self.dropout_p, # Pass dropout_p here
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)
            out_channels = block_out_channels # Update out_channels for the next block's input

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        if self.use_final_do:
            self.final_dropout = nn.Dropout(p=self.final_dropout_p)
        self.dense = nn.Linear(out_channels, self.out_dim) # Use self.out_dim
        # Removed dense2 as backbone logic handles feature output

    def forward(self, x):
        # Input expected: (batch_size, sequence_length, num_features/channels)
        # Transpose to: (batch_size, num_features/channels, sequence_length) for Conv1d
        x = x.transpose(1, 2)
        out = x

        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)

        # residual blocks
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print(f'i_block: {i_block}, in_channels: {net.in_channels}, out_channels: {net.out_channels}, downsample: {net.downsample}')
            out = net(out)
            if self.verbose:
                print(f'Block {i_block} output shape: {out.shape}')

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out) # Apply final activation

        # Global Average Pooling
        out = out.mean(dim=-1) # Pool across the sequence length dimension
        if self.verbose:
            print('final pooling', out.shape)

        if self.use_final_do:
            out = self.final_dropout(out)

        # If backbone mode, return features before the final dense layer
        if self.backbone:
            # Note: The original code returned None, out. Returning just 'out' (features) is more typical.
            # Also removed self.dense2 as it seemed redundant if backbone just needs features.
            return out # Return pooled features

        # Final dense layer for prediction
        out_value = self.dense(out)
        if self.verbose:
            print('dense output', out_value.shape)

        # Squeeze output if it's single-value regression (output_dim=1)
        if self.out_dim == 1:
            out_value = out_value.squeeze(-1)
            if self.verbose:
                print('squeeze output', out_value.shape)

        return out_value, out
