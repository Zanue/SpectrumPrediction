from torch import nn
import numpy as np

class MlpBlock(nn.Module):
    def __init__(self, input_dim, output_dim, middle=None):
        super(MlpBlock, self).__init__()
        if middle is None:
            middle =  (input_dim + output_dim) // 2
        self.fc1 = nn.Linear(input_dim, middle)
        self.fc2 = nn.Linear(middle, output_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self, in_tokens, out_tokens, tokens_dim, middle=None):
        super(MixerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_tokens)
        self.mlp_token_mixing = MlpBlock(tokens_dim, tokens_dim)
        self.norm2 = nn.LayerNorm(in_tokens)
        self.mlp_channel_mixing = MlpBlock(in_tokens, out_tokens, middle)
        self.skip = nn.Linear(in_tokens, out_tokens) if in_tokens != out_tokens else nn.Identity()

    def forward(self, x):
        y = self.norm1(x) #[B, C, L]
        y = y.permute(0,2,1) #[B, L, C]
        y = self.mlp_token_mixing(y) #[B, L, C]
        y = y.permute(0,2,1) #[B, C, L]
        x = x + y
        y = self.norm2(x) #[B, C, L]
        y = self.skip(x) + self.mlp_channel_mixing(y) #[B, C, L]

        return y


class MLPMixer(nn.Module):
    def __init__(self, input_len, output_len, num_channels, num_blocks, middle=None):
        super(MLPMixer, self).__init__()

        # middle_layers = np.linspace(input_len, output_len, num_blocks+1, dtype=np.int)
        blocks = []
        for i in range(num_blocks):
            blocks.append(MixerBlock(input_len, input_len, num_channels, middle))
        self.blocks = nn.Sequential(*blocks)
        # self.norm = nn.LayerNorm(input_len)
        self.fc = nn.Linear(input_len, output_len)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.permute(0,2,1) #[B, C, L]
        y = self.blocks(x) #[B, C, L]
        # y = self.norm(y)
        y = self.fc(y) #[B, C, L]

        return y.permute(0,2,1)