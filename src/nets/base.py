# import torch
# import torch.nn as nn
import paddle
import paddle.nn as nn

class Residual_Block(nn.Layer):
    def __init__(self, n_feats=64, res_scale=0.1):
        super(Residual_Block, self).__init__()

        self.conv1 = nn.Conv2D(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2D(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.res_scale = res_scale

    def forward(self, x):
        output = self.conv2(self.relu(self.conv1(x)))
        output = output*self.res_scale
        # output = output.mul(self.res_scale)
        output += x
        return output

class Residual_Block_Dilate(nn.Layer):
    def __init__(self, n_feats=64, res_scale=0.1):
        super(Residual_Block_Dilate, self).__init__()

        self.conv1 = nn.Conv2D(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2D(n_feats, n_feats, kernel_size=3, stride=1, padding=2, dilation=2, bias_attr=False)
        self.res_scale = res_scale

    def forward(self, x):
        output = self.conv2(self.relu(self.conv1(x)))
        output = output.mul(self.res_scale)
        output += x
        return output
