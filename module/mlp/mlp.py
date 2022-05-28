from torch import nn

class MLP(nn.Module):
    def __init__(self, input_len, output_len, middle=None):
        super(MLP, self).__init__()

        if middle is None:
            middle = output_len
        self.linear1 = nn.Sequential(
            nn.Linear(input_len, middle),
            nn.ReLU(),
            nn.Linear(middle, middle)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(middle, middle),
            nn.ReLU(),
            nn.Linear(middle, output_len)
        )

        self.skip = nn.Linear(input_len, middle)
        self.act = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.05)

    def forward(self, x):
        x = x.permute(0, 2, 1) # [B, C, L]
        y1 = self.act(self.linear1(x) + self.skip(x))
        y2 = self.linear2(y1)

        return y2.permute(0,2,1) # [B, L, C]
