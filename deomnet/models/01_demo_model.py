import torch
import torch.nn as nn
import torch.quantization
import brevitas.nn as qnn
import brevitas.onnx as bo
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantAvgPool2d, QuantIdentity
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
 
 
def Conv3x3(in_channels, out_channels):
    return QuantConv2d(in_channels, 
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    weight_quant=CommonIntWeightPerChannelQuant,
                    weight_bit_width=8,
                    return_quant_tensor=True)

# conv + bn + relu
class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            weight_bit_width,
            act_bit_width,
            stride=1,
            padding=0,
            groups=1,
            bn_eps=1e-5,
            activation_scaling_per_channel=False):
        super(ConvBlock, self).__init__()
        self.conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=weight_bit_width,
            return_quant_tensor=True)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activation = QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            per_channel_broadcastable_shape=(1, out_channels, 1, 1),
            scaling_per_channel=activation_scaling_per_channel,
            return_quant_tensor=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# conv + bn + relu + conv + bn + (residual)relu
class BasicBlock(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=8, act_quant=CommonUintActQuant, return_quant_tensor=True)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=8,
            return_quant_tensor=True)
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out =  self.quant_inp(out)
        residual= self.quant_inp(x)
        out = out + residual
        out = self.relu(out)
 
        return out

# dwconv + pwconv
# dwconv = conv + bn + relu  pwconv = conv + bn + relu
class DwsConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            stride,
            bit_width,
            pw_activation_scaling_per_channel=False):
        super(DwsConvBlock, self).__init__()
        self.dw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            weight_bit_width=bit_width,
            act_bit_width=bit_width)
        self.pw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            weight_bit_width=bit_width,
            act_bit_width=bit_width,
            activation_scaling_per_channel=pw_activation_scaling_per_channel)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x



class DemoNet(nn.Module):
    def __init__(self):
        super(DemoNet, self).__init__()
    
        # Identity input
        #self.quant_inp = QuantIdentity(bit_width=8, return_quant_tensor=True,)

        # init part
        self.init_block = ConvBlock(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=0,
            weight_bit_width=FIRST_LAYER_BIT_WIDTH,
            activation_scaling_per_channel=False,
            act_bit_width=8)

        # dwsconv part
        self.features = nn.Sequential()
        dwsstage = nn.Sequential()    
        dwsconv1 = DwsConvBlock(
                    in_channels=8,
                    out_channels=16,
                    stride=1,
                    bit_width=8,
                    pw_activation_scaling_per_channel=False)
        dwsstage.add_module('dwconv1', dwsconv1)
        self.features.add_module('dwsstage', dwsstage)

        # residual part
        self.resblock =  nn.Sequential(BasicBlock(16, 16))

        # last conv or linear part
        self.lastblock = nn.Sequential(QuantConv2d(16, 64, kernel_size=3, stride=1,bias=False))

    def forward(self, x):
        #x = self.quant_inp(x)
        x = self.init_block(x)
        x = self.features(x)  
        x = self.resblock(x)
        out = self.lastblock(x)

        return out


model_for_export = "demonet.pth"
ready_model_filename = "demonet-ready.onnx"

model = DemoNet()
print(model)
torch.save(model.state_dict(),model_for_export)
input_shape = (1, 1, 224, 224)
bo.export_finn_onnx(model, input_shape, export_path=ready_model_filename)
print("Model saved to %s" % ready_model_filename)