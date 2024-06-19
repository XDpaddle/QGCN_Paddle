# import torch
# import torch.nn as nn
import paddle
import paddle.nn as nn

import math
from nets import base

class Net(nn.Layer):
    def __init__(self, in_channel, out_channel, n_colors=3, n_feats=64, n_resblocks=16, res_scale=0.1, scale=2):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2D(in_channel, n_feats, kernel_size=3, stride=1, padding=1, bias_attr=False)

        self.downscale = nn.Sequential(
            nn.Conv2D(n_feats, n_feats, kernel_size=4, stride=2, padding=1, bias_attr=False),
            nn.ReLU(True)
        )
        
        residual = [
            base.Residual_Block(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks)
        ]
        self.residual = nn.Sequential(*residual)

        self.conv_mid = nn.Conv2D(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias_attr=False)

        self.upscale = nn.Sequential(
            nn.Conv2D(n_feats, (scale**2)*n_feats, kernel_size=3, stride=1, padding=1, bias_attr=True),
            nn.PixelShuffle(scale),
        )

        self.conv_output = nn.Conv2D(n_feats, out_channel, kernel_size=3, stride=1, padding=1, bias_attr=False)
        
        for m in self.sublayers():

            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # if m.bias is not None:
                #     m.bias.data.zero_()
                m.weight.set_value(paddle.randn(shape=m.weight.shape) * math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.set_value(paddle.zeros(shape=m.bias.shape))

    def forward(self, x):
        # print(x.size())
        x = paddle.to_tensor(x, dtype='float32')
        out = self.conv_input(x)
        skip = out
        out = self.downscale(out)
        out = self.residual(out)
        out = self.conv_mid(out)
        out = self.upscale(out)
        out += skip
        out = self.conv_output(out)
        return out
