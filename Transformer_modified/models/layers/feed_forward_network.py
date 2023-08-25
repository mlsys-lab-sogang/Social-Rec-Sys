import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, ffn_size, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(d_model, ffn_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer2 = nn.Linear(ffn_size, d_model)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)

        return x