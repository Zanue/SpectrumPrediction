from torch import nn
from .tcn import TemporalConvNet

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout,seq_length,pred_length):
        super(TCN, self).__init__()
        self.input_size = input_size

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.conv_1_1 = nn.Conv1d(num_channels[-1],input_size,1,1,0)
        self.linear = nn.Linear(seq_length, pred_length)  #channel to be 1

        self.init_weights()
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.01)
        self.linear.weight.data.normal_(0, 0.01)


    def forward(self, x):
        y = self.tcn(x.permute(0,2,1))  # (B,D,L_in)
        y = self.conv_1_1(y) #(B,C,L_in)
        y = self.linear(y)   #(B,C,L_out)
        
        return y.permute(0,2,1) # (B,L,C)