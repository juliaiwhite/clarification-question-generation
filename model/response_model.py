import torch.nn as nn

class ResponseModel(nn.Module):
    def __init__(self, out_channels = 1):
        super(ResponseModel, self).__init__()
        self.block1 = self.conv_block(c_in=1024, c_out=512, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=512, c_out=256, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block4 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv1d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.lastcnn(x)
        x = x.squeeze(1)
        return x
    
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm1d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        return seq_block