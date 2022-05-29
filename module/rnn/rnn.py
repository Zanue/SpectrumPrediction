from torch import nn

class RNNet2(nn.Module):
    def __init__(self,num_layers,seq_len,pred_len):
        super(RNNet2, self).__init__()
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.num_layers=num_layers
        self.rnn=nn.RNN(
            input_size=2048,
            hidden_size=2048*int(self.pred_len/self.seq_len),
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.linear=nn.Sequential(
            nn.Linear(in_features=2048*int(self.pred_len/self.seq_len),out_features=2048*int(self.pred_len/self.seq_len))
        )

    def forward(self, x):
    # 输入的x的形状为(batch_size,seq_len,2048)
        out1,hidden = self.rnn(x) #out1形状为(batch_size,seq_len,hidden_size)
        out2 = self.linear(out1)  #out2形状为(batch_size,seq_len,hidden_size)
        output=out2.view(-1,self.pred_len,2048)

        return output #output形状为(batch_size,pred_len,2048)

class RNNet(nn.Module):
    def __init__(self,num_layers,seq_len,pred_len):
        super(RNNet, self).__init__()
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.num_layers=num_layers
        self.rnn=nn.RNN(
            input_size=2048,
            hidden_size=2048,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.linear=nn.Sequential(
            nn.Linear(in_features=self.seq_len,out_features=self.pred_len)
        )

    def forward(self, x):
    # 输入的x的形状为(batch_size,seq_len,2048)
        out1,hidden = self.rnn(x) #out1形状为(batch_size,seq_len,hidden_size)
        out2 = self.linear(out1.permute(0,2,1))  #out2形状为(batch_size,seq_len,hidden_size)
        output=out2.permute(0,2,1)

        return output #output形状为(batch_size,pred_len,2048)