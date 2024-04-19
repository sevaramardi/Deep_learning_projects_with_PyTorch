import torch
import torch.nn as nn
import torch.nn.functional as F



class ChannelAttention(nn.Module):
    def __init__(self, kernel_size=7): 
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3,7), 'kernel size has to be 3 or 7'
        padding = 3 if kernel_size ==7 else 1

        self.conv1 = nn.Conv2d(2,1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x = 8x8
        avg_out = torch.mean(x, dim=1, keepdim=True) # 1,8x8
        max_out, _ = torch.max(x, dim=1, keepdim=True) #1,8x8
        x = torch.cat([avg_out, max_out], dim=1) #2,8x8
        x = self.conv1(x) #1,8x8
        return self.sigmoid(x)
    
spatialAttention = SpatialAttention()
