import torch.nn as nn
class SKAttention(nn.Module):
    def __init__(self, channel, reduction):
        super(SKAttention, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=2, dilation=2, bias=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_se = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_ex = nn.Sequential(nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True))
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        conv1 = self.conv1(x).unsqueeze(dim=1)
        conv2 = self.conv2(x).unsqueeze(dim=1)
        features = torch.cat([conv1, conv2], dim=1)
        U = torch.sum(features, dim=1)
        S = self.pool(U)
        Z = self.conv_se(S)
        attention_vector = torch.cat([self.conv_ex(Z).unsqueeze(dim=1), self.conv_ex(Z.unsqueeze(dim=1)], dim=1)
        attention_vector = self.softmax(attention_vector)
        V = (features * attention_vector).sum(dim=1)
        return V