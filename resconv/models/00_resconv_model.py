import torch
import torch.nn as nn
import torch.quantization
import brevitas.nn as qnn
import brevitas.onnx as bo
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantAvgPool2d
from brevitas.quant import IntBias

from brevitas.core.restrict_val import RestrictValueType
from brevitas.quant import Uint8ActPerTensorFloatMaxInit, Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerTensorFloat



 
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = QuantReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class ResConv(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(ResConv, self).__init__()

        self.init = nn.Sequential(
            QuantConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            QuantReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block0 =  nn.Sequential(BasicBlock(64, 64, 1, None))
        self.block1 =  nn.Sequential(BasicBlock(64, 64, 1, None))

    def forward(self, x):
        x = self.init(x)
        x = self.block0(x)
        x = self.block1(x)

        return x


model_for_export = "resconv_ori.pth"
ready_model_filename = "resconv_ori-ready.onnx"

model = ResConv()
print(model)
torch.save(model.state_dict(),model_for_export)
input_shape = (1, 3, 224, 224)
bo.export_finn_onnx(model, input_shape, export_path=ready_model_filename,)
print("Model saved to %s" % ready_model_filename)