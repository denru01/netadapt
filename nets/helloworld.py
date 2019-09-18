import torch
import torch.nn as nn


__all__ = ['HelloWorld', 'helloworld']

class HelloWorld(nn.Module):

    def __init__(self, num_classes=10):
        super(HelloWorld, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.avgpool = nn.AvgPool2d(32, 32)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x

def helloworld(num_classes=10):
    return HelloWorld()
    