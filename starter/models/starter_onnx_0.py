from pickle import FALSE
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

class Depth_Conv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            bit_width,
            act_bit_width,
            stride=1,
            padding=0,
            groups=1,
            pw_activation_scaling_per_channel=False):
        super(Depth_Conv2d, self).__init__()
        self.dw_conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=bit_width)
        self.activation = QuantReLU(
            act_quant=CommonUintActQuant,
            bit_width=act_bit_width,
            per_channel_broadcastable_shape=(1, out_channels, 1, 1),
            scaling_per_channel=pw_activation_scaling_per_channel,
            return_quant_tensor=True)
        self.pw_conv = QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            weight_quant=CommonIntWeightPerChannelQuant,
            weight_bit_width=bit_width)
    
    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x

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
        self.quant_inp = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)

        self.stage1 = nn.Sequential()
        init_block = ConvBlock(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=2,
            weight_bit_width=FIRST_LAYER_BIT_WIDTH,
            activation_scaling_per_channel=True,
            act_bit_width=8)
        
        conv2 = DwsConvBlock(
                    in_channels=8,
                    out_channels=16,
                    stride=1,
                    bit_width=8,
                    pw_activation_scaling_per_channel=True)  
        conv3 = DwsConvBlock(
                    in_channels=16,
                    out_channels=16,
                    stride=2,
                    bit_width=8,
                    pw_activation_scaling_per_channel=True)
        conv4 = DwsConvBlock(
                    in_channels=16,
                    out_channels=16,
                    stride=1,
                    bit_width=8,
                    pw_activation_scaling_per_channel=True)
        #self.conv5 = conv_dw(16, 32, 2)
        conv5 = DwsConvBlock(
                    in_channels=16,
                    out_channels=32,
                    stride=2,
                    bit_width=8,
                    pw_activation_scaling_per_channel=True)
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
                    pw_activation_scaling_per_channel=True)
        
        self.stage1.add_module('init_block', init_block)
        self.stage1.add_module('dwconv2', conv2)
        self.stage1.add_module('dwconv3', conv3)
        self.stage1.add_module('dwconv4', conv4)
        self.stage1.add_module('dwconv5', conv5)
        self.stage1.add_module('dwconv6', conv6)
        self.stage1.add_module('dwconv7', conv7)
        self.stage1.add_module('dwconv8', conv8)
        
        self.stage2 = nn.Sequential()
        #self.conv9 = conv_dw(32, 64, 2)
        conv9 = DwsConvBlock(
                    in_channels=32,
                    out_channels=64,
                    stride=2,
                    bit_width=8,
                    pw_activation_scaling_per_channel=True)
        #self.conv10 = conv_dw(64, 64, 1)
        conv10 = DwsConvBlock(
                    in_channels=64,
                    out_channels=64,
                    stride=1,
                    bit_width=8,
                    pw_activation_scaling_per_channel=False)
        #self.conv11 = conv_dw(64, 64, 1)
        conv11 = DwsConvBlock(
                    in_channels=64,
                    out_channels=64,
                    stride=1,
                    bit_width=8,
                    pw_activation_scaling_per_channel=False)

        self.stage2.add_module('dwconv9', conv9)
        self.stage2.add_module('dwconv10', conv10)
        self.stage2.add_module('dwconv11', conv11)


        self.stage3 = nn.Sequential()
        #self.conv12 = conv_dw(64, 128, 2)
        conv12 = DwsConvBlock(
                    in_channels=64,
                    out_channels=128,
                    stride=2,
                    bit_width=8,
                    pw_activation_scaling_per_channel=True)
        #self.conv13 = conv_dw(128, 128, 1)
        conv13 = DwsConvBlock(
                    in_channels=128,
                    out_channels=128,
                    stride=1,
                    bit_width=8,
                    pw_activation_scaling_per_channel=True)

        self.stage3.add_module('conv12', conv12)
        self.stage3.add_module('conv13', conv13)

        self.stage4 = nn.Sequential()
        conv14 = nn.Sequential(
            QuantConv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=1,
                bias=False,
                weight_quant=CommonIntWeightPerTensorQuant,
                weight_bit_width=8,
                return_quant_tensor=True),
            QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=8,
                per_channel_broadcastable_shape=(1, 32, 1, 1),
                scaling_per_channel=True,
                return_quant_tensor=True),
            QuantConv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=32,
                bias=False,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=8,
                return_quant_tensor=True),
            QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=8,
                per_channel_broadcastable_shape=(1, 32, 1, 1),
                scaling_per_channel=True,
                return_quant_tensor=True),
            QuantConv2d(
                in_channels=32,
                out_channels=128,
                kernel_size=1,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=8,
                return_quant_tensor=True),
            QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=8,
                per_channel_broadcastable_shape=(1, 128, 1, 1),
                scaling_per_channel=True,
                return_quant_tensor=True),
        )
        self.stage4.add_module('conv14', conv14)

        self.loc_layer1 = nn.Sequential(
        #loc_layers += [depth_conv2d(32, 12, kernel=3, pad=1)]
            QuantConv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=32,
                bias=False,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=8),
            QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=8,
                per_channel_broadcastable_shape=(1, 32, 1, 1),
                scaling_per_channel=False,
                return_quant_tensor=True),
            QuantConv2d(
                in_channels=32,
                out_channels=12,
                kernel_size=1,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=8,
		        return_quant_tensor=False),
        )
        
        """
        #loc_layers += [depth_conv2d(64, 8, kernel=3, pad=1)]
        self.loc_layer2 = nn.Sequential(
            QuantConv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=64,
                bias=False,
		#input_quant=CommonIntWeightPerTensorQuant,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=8,
                return_quant_tensor=True),
            QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=8,
                per_channel_broadcastable_shape=(1, 64, 1, 1),
                scaling_per_channel=False,
                return_quant_tensor=True),
            QuantConv2d(
                in_channels=64,
                out_channels=8,
                kernel_size=1,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=8,
		return_quant_tensor=True),
	    )


        #loc_layers += [depth_conv2d(128, 8, kernel=3, pad=1)]
        self.loc_layer3 = nn.Sequential(
            QuantConv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=128,
                bias=False,
		#input_quant=CommonIntWeightPerTensorQuant,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=8,
		return_quant_tensor=True),
            QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=8,
                per_channel_broadcastable_shape=(1, 128, 1, 1),
                scaling_per_channel=True,
                return_quant_tensor=False),
            QuantConv2d(
                in_channels=128,
                out_channels=8,
                kernel_size=1,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=8),
        )
        #loc_layers += [qnn.QuantConv2d(128, 12, kernel_size=3, padding=1, weight_bit_width=8)]
        self.loc_layer4 = nn.Sequential(
            QuantConv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=128,
                bias=False,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=8,
		return_quant_tensor=True),
            QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=8,
                per_channel_broadcastable_shape=(1, 128, 1, 1),
                scaling_per_channel=True,
                return_quant_tensor=True),
            QuantConv2d(
                in_channels=128,
                out_channels=12,
                kernel_size=1,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=8),
        )
	"""

        #self.conf_layer = nn.Sequential()
        #conf_layers += [depth_conv2d(32, 6, kernel=3, pad=1)]
        #conf_layers += [depth_conv2d(64, 4, kernel=3, pad=1)]
        #conf_layers += [depth_conv2d(128, 4, kernel=3, pad=1)]
        #conf_layers += [qnn.QuantConv2d(128, 6, kernel_size=3, padding=1, weight_bit_width=8)]
        
        
        #self.landm_layer = nn.Sequential()
        #landm_layers += [depth_conv2d(32, 30, kernel=3, pad=1)]
        #landm_layers += [depth_conv2d(64, 20, kernel=3, pad=1)]
        #landm_layers += [depth_conv2d(128, 20, kernel=3, pad=1)]
        #landm_layers += [qnn.QuantConv2d(128, 30, kernel_size=3, padding=1, weight_bit_width=8)]


    def forward(self,inputs):
        s0 = self.quant_inp(inputs)
        s1 = self.stage1(s0)
        q1 = self.quant_inp(s1)
        s2 = self.stage2(q1)
        #s3 = self.stage3(s2)
        #s4 = self.stage4(s3)

        #q2 = self.quant_inp(s2)
        #q3 = self.stage3(s2)
        #q4 = self.stage4(s3)
        loc = list()
        loc.append(self.loc_layer1(q1))
        #loc.append(self.loc_layer2(q2))
        #loc.append(self.loc_layer3(q3))
        #loc.append(self.loc_layer4(q4))

        #output = torch.cat([o.view(o.size(0), -1, 4)  for o in loc], 1)
        return loc, s2

model = Starter()
model_for_export = "starter_qnn.pth"
print(model)
torch.save(model.state_dict(),model_for_export)

ready_model_filename = "starter_qnn-ready.onnx"
input_shape = (1, 1, 320, 320)
bo.export_finn_onnx(model, input_shape, export_path=ready_model_filename,)
print("Model saved to %s" % ready_model_filename)
