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


class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None


class CommonIntWeightPerChannelQuant(CommonIntWeightPerTensorQuant):
    """
    Common per-channel weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_per_output_channel = True


class CommonIntActQuant(Int8ActPerTensorFloatMinMaxInit):
    """
    Common signed act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    min_val = -10.0
    max_val = 10.0
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonUintActQuant(Uint8ActPerTensorFloatMaxInit):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    max_val = 6.0
    restrict_scaling_type = RestrictValueType.LOG_FP

FIRST_LAYER_BIT_WIDTH = 8
 
 
def conv3x3(in_planes, out_planes, stride=1, weight_bit_width=8):
    """3x3 convolution with padding"""
    return QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False,weight_quant=CommonIntWeightPerTensorQuant,
                    weight_bit_width=weight_bit_width)

class BasicBlock(nn.Module): 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True, act_quant=CommonUintActQuant)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.relu = self.relu = QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=8,
            return_quant_tensor=True)
            
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out =  self.quant_inp(out)
        residual= self.quant_inp(x)
        out += residual
        out = self.relu(out)
 
        return out

class ResConv(nn.Module):
    def __init__(self):
        super(ResConv, self).__init__()
        
        #self.quant_inp = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True, act_quant=CommonUintActQuant)
        self.init = nn.Sequential(
            QuantConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=8,
                return_quant_tensor=True),
            QuantAvgPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.block =  nn.Sequential(BasicBlock(64, 64, 1, None))
        self.last = nn.Sequential(QuantConv2d(64, 128, kernel_size=3, stride=1,bias=False))

    def forward(self, x):
        #x = self.quant_inp(x)
        x = self.init(x)
        x = self.block(x)
        x = self.last(x)

        return x


model_for_export = "resconv.pth"
ready_model_filename = "resconv-ready.onnx"

model = ResConv()
print(model)
torch.save(model.state_dict(),model_for_export)
input_shape = (1, 3, 224, 224)
bo.export_finn_onnx(model, input_shape, export_path=ready_model_filename,)
print("Model saved to %s" % ready_model_filename)