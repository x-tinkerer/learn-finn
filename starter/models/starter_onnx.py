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

"""
class QuantConv1d(QuantWBIOL, Conv1d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_type: str = 'standard',
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
"""


def depth_conv2d(inp, oup, kernel=1, stride=1, pad=0):
    return nn.Sequential(
        qnn.QuantConv2d(inp, inp, kernel_size = kernel, stride = stride, padding=pad, groups=inp, weight_bit_width=8, return_quant_tensor=True),
        qnn.QuantReLU(bit_width=8, return_quant_tensor=True),
        qnn.QuantConv2d(inp, oup, kernel_size=1, weight_bit_width=8, return_quant_tensor=True)
    )

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
            weight_bit_width=weight_bit_width)
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

"""
TODO:
pw_activation_scaling_per_channel=True?

AssertionError: cycle-free graph violated: partition depends on itself
"""
class Starter(nn.Module):
    def __init__(self):
        super(Starter, self).__init__()
        #self.quant_inp = qnn.QuantIdentity(bit_width=8)
        
        self.features = nn.Sequential()
        
        init_block = ConvBlock(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=2,
            weight_bit_width=FIRST_LAYER_BIT_WIDTH,
            activation_scaling_per_channel=True,
            act_bit_width=8)
        
        self.features.add_module('init_block', init_block)

        
        stage = nn.Sequential()
        #self.conv2 = conv_dw(8, 16, 1)
        conv2 = DwsConvBlock(
                    in_channels=8,
                    out_channels=16,
                    stride=1,
                    bit_width=8,
                    pw_activation_scaling_per_channel=True)
        
        #self.conv3 = conv_dw(16, 16, 2)
        conv3 = DwsConvBlock(
                    in_channels=16,
                    out_channels=16,
                    stride=2,
                    bit_width=8,
                    pw_activation_scaling_per_channel=True)
        
        #self.conv4 = conv_dw(16, 16, 1)
        conv4 = DwsConvBlock(
                    in_channels=16,
                    out_channels=16,
                    stride=1,
                    bit_width=8,
                    pw_activation_scaling_per_channel=False)
        
        """
        #self.conv5 = conv_dw(16, 32, 2)
        conv5 = DwsConvBlock(
                    in_channels=16,
                    out_channels=32,
                    stride=2,
                    bit_width=8,
                    pw_activation_scaling_per_channel=False)
        
        #self.conv6 = conv_dw(32, 32, 1)
        conv6 = DwsConvBlock(
                    in_channels=32,
                    out_channels=32,
                    stride=1,
                    bit_width=8,
                    pw_activation_scaling_per_channel=True)
        #self.conv7 = conv_dw(32, 32, 1)
        conv7 = DwsConvBlock(
                    in_channels=32,
                    out_channels=32,
                    stride=1,
                    bit_width=8,
                    pw_activation_scaling_per_channel=True)
        #self.conv8 = conv_dw(32, 32, 1)
        conv8 = DwsConvBlock(
                    in_channels=32,
                    out_channels=32,
                    stride=1,
                    bit_width=8,
                    pw_activation_scaling_per_channel=False)
        """

        stage.add_module('dwconv2', conv2)
        stage.add_module('dwconv3', conv3)
        stage.add_module('dwconv4', conv4)
        #stage.add_module('dwconv5', conv5)
        #stage.add_module('dwconv6', conv6)
        #stage.add_module('dwconv7', conv7)
        #stage.add_module('dwconv8', conv8)
        self.features.add_module('stage1', stage)
        
        self.final_pool = QuantAvgPool2d(kernel_size=7, stride=1, bit_width=8)

        self.output = QuantLinear(
            1600, 10,
            bias=True,
            bias_quant=IntBias,
            weight_quant=CommonIntWeightPerTensorQuant,
            weight_bit_width=8)

    def forward(self,inputs):
        #x = self.quant_inp(inputs)
        x = self.features(inputs)
        x =self.final_pool(x)
        x = x.view(x.size(0), -1)
        output = self.output(x)
        return output


model = Starter()
model_for_export = "starter_qnn.pth"
torch.save(model.state_dict(),model_for_export)

ready_model_filename = "starter_qnn-ready.onnx"
input_shape = (1, 1, 64, 64)

bo.export_finn_onnx(model, input_shape, export_path=ready_model_filename)

print("Model saved to %s" % ready_model_filename)
