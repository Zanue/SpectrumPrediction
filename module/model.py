from torch import nn
from .tcn import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout,seq_length,pred_length):
        super(TCN, self).__init__()
        self.input_size = input_size

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.conv_1_1 = nn.Conv1d(num_channels[-1],input_size,1,1,0)
        self.linear = nn.Linear(seq_length*seq_length, pred_length)  #channel to be 1

        self.init_weights()
        

    def init_weights(self):
        if self.use_ConvDown:
            for m in self.downSample.modules():
                if isinstance(m, nn.Conv1d):
                    m.weight.data.normal_(0, 0.01)
        else:
            self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)  #y1 (B,C,L,L)
        B,C,_,_ = y1.shape
        y2 = y1.reshape(B,C,-1)
        y3 = self.conv_1_1(y2)
        y4 = self.linear(y3)   #(B,C1,L)
        
        return y4.permute(0,2,1) #输出 （B,L,C）
        

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
        self.act = nn.ReLU()

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

        