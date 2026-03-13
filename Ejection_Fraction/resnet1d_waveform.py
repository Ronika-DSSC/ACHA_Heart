import torch
import torch.nn as nn
from typing import Optional, Type


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dropout: float = 0.5,
        kernel_size: int = 7,
        padding: int = 3,
        bias: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size, 1, padding, bias=bias)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class ResNet1dWaveform(nn.Module):
    def __init__(self, filter_size=64, input_channels=12, num_classes=1):
        super().__init__()

        self.inplanes = filter_size
        layers = [3, 4, 6, 3]

        self.conv1 = nn.Conv1d(input_channels, filter_size, 15, 2, 7)
        self.bn1 = nn.BatchNorm1d(filter_size)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        self.layer1 = self._make_layer(BasicBlock1d, filter_size, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 2 * filter_size, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 4 * filter_size, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 8 * filter_size, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(8 * filter_size * 2, num_classes)

    def _make_layer(self, block: Type[nn.Module], planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Expect input shape: [B, 12, 5000]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.avgpool(x)
        x2 = self.maxpool2(x)

        x = torch.cat([x1, x2], dim=1)
        x = x.view(x.size(0), -1)

        return self.fc(x).squeeze(1)