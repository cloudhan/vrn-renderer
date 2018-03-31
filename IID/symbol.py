import mxnet as mx
import numpy as np


Var  = mx.sym.Variable
Conv = mx.sym.Convolution
BN   = mx.sym.BatchNorm
Acti = mx.sym.Activation
Deconv = mx.sym.Deconvolution


def ConvBlockFactory(name_prefix, data, kernel, num_filter, stride, pad, bn_eps=2e-5, bn_fix_gamma=False, act_type="relu"):
    conv = Conv(data=data, kernel=kernel, num_filter=num_filter, stride=stride, pad=pad, name=name_prefix+"_conv")
    bn   = BN(data=conv, eps=bn_eps, fix_gamma=bn_fix_gamma, name=name_prefix+"_bn",)
    if act_type=="relu":
        acti = Acti(data=bn, act_type=act_type, name=name_prefix+"_"+act_type)
    elif act_type=="leaky":
        acti = mx.sym.LeakyReLU(data=bn, act_type=act_type, name=name_prefix+"_"+act_type, slope=0.1)
    return acti

def generator_symbol():
    use_depth=False
    act_type="leaky"

    image = Var(name="image")
    mask = Var(name="mask")  # 0 or 255.0
    mask01 = mask / 255.0

    if use_depth:
        depth = Var(name="noisy_depth")
        data = mx.sym.Concat(image, mask, depth)
    else:
        data = mx.sym.Concat(image, mask)
    down0 =   ConvBlockFactory("down0",   data=data, kernel=(3,3), num_filter=48, stride=(1,1), pad=(1,1), act_type=act_type)

    # down0to1 = mx.sym.Pooling(name="down0to1", data=down0, kernel=(2,2), stride=(2,2), pool_type="max")
    down0to1 = ConvBlockFactory("down0to1", data=down0, kernel=(3,3), num_filter=48, stride=(2,2), pad=(1,1), act_type=act_type)
    down1 = ConvBlockFactory("down1", data=down0to1, kernel=(3,3), num_filter=64, stride=(1,1), pad=(1,1), act_type=act_type)

    # down1to2 = mx.sym.Pooling(name="down1to2", data=down1, kernel=(2,2), stride=(2,2), pool_type="max")
    down1to2 = ConvBlockFactory("down1to2", data=down1, kernel=(3,3), num_filter=64, stride=(2,2), pad=(1,1), act_type=act_type)
    down2 = ConvBlockFactory("down2", data=down1to2, kernel=(3,3), num_filter=128, stride=(1,1), pad=(1,1), act_type=act_type)

    # down2to3 = mx.sym.Pooling(name="down2to3", data=down2, kernel=(2,2), stride=(2,2), pool_type="max")
    down2to3 = ConvBlockFactory("down2to3", data=down2, kernel=(3,3), num_filter=128, stride=(2,2), pad=(1,1), act_type=act_type)
    down3 = ConvBlockFactory("down3", data=down2to3, kernel=(3,3), num_filter=256, stride=(1,1), pad=(1,1), act_type=act_type)

    # down3to4 = mx.sym.Pooling(name="down3to4", data=down3, kernel=(2,2), stride=(2,2), pool_type="max")
    down3to4 = ConvBlockFactory("down3to4", data=down3, kernel=(3,3), num_filter=256, stride=(2,2), pad=(1,1), act_type=act_type)
    down4 = ConvBlockFactory("down4", data=down3to4, kernel=(3,3), num_filter=512, stride=(1,1), pad=(1,1), act_type=act_type)
    up4to3 = Deconv(name="up4to3", data=down4, kernel=(4,4), num_filter=512, stride=(2,2), pad=(1,1))

    up3concat = mx.sym.Concat(down3, up4to3, name="up3concat")
    up3 = ConvBlockFactory("up3", data=up3concat, kernel=(3,3), num_filter=256, stride=(1,1), pad=(1,1), act_type=act_type)
    up3to2 = Deconv(name="up3to2", data=up3, kernel=(4,4), num_filter=256, stride=(2,2), pad=(1,1))

    up2concat = mx.sym.Concat(down2, up3to2, name="up2concat")
    up2 = ConvBlockFactory("up2_1", data=up2concat, kernel=(3,3), num_filter=128, stride=(1,1), pad=(1,1), act_type=act_type)
    up2to1 = Deconv(name="up2to1", data=up2, kernel=(4,4), num_filter=128, stride=(2,2), pad=(1,1))

    up1concat = mx.sym.Concat(down1, up2to1, name="up1concat")
    up1 = ConvBlockFactory("up1", data=up1concat, kernel=(3,3), num_filter=64, stride=(1,1), pad=(1,1), act_type=act_type)
    up1to0 = Deconv(name="up1to0", data=up1, kernel=(4,4), num_filter=64, stride=(2,2), pad=(1,1))

    up0concat = mx.sym.Concat(down0, up1to0, name="up0concat")

    up0_1 = ConvBlockFactory("up0_1", data=up0concat, kernel=(3,3), num_filter=72, stride=(1,1), pad=(1,1), act_type=act_type)
    up0_2 = ConvBlockFactory("up0_2", data=up0_1, kernel=(3,3), num_filter=72, stride=(1,1), pad=(1,1), act_type=act_type)

    decode1 = ConvBlockFactory("decode1", data=up0_2, kernel=(3,3), num_filter=64, stride=(1,1), pad=(1,1), act_type=act_type)
    decode2 = ConvBlockFactory("decode2", data=decode1, kernel=(3,3), num_filter=48, stride=(1,1), pad=(1,1), act_type=act_type)
    decode3 = ConvBlockFactory("decode3", data=decode2, kernel=(3,3), num_filter=48, stride=(1,1), pad=(1,1), act_type=act_type)
    pred = Conv(name="pred_conv", data=decode3, kernel=(1,1), num_filter=3, stride=(1,1), pad=(0,0))
    pred = mx.sym.broadcast_mul(pred, mask01)
    return pred




def descriptor_symbol(style_layers, content_layers):
    data = mx.symbol.Variable('data')
    conv1_1 = mx.symbol.Convolution(name='conv1_1', data=data , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_1 = mx.symbol.Activation(name='relu1_1', data=conv1_1 , act_type='relu')
    conv1_2 = mx.symbol.Convolution(name='conv1_2', data=relu1_1 , num_filter=64, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu1_2 = mx.symbol.Activation(name='relu1_2', data=conv1_2 , act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu1_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv2_1 = mx.symbol.Convolution(name='conv2_1', data=pool1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_1 = mx.symbol.Activation(name='relu2_1', data=conv2_1 , act_type='relu')
    conv2_2 = mx.symbol.Convolution(name='conv2_2', data=relu2_1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu2_2 = mx.symbol.Activation(name='relu2_2', data=conv2_2 , act_type='relu')
    pool2 = mx.symbol.Pooling(name='pool2', data=relu2_2 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv3_1 = mx.symbol.Convolution(name='conv3_1', data=pool2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_1 = mx.symbol.Activation(name='relu3_1', data=conv3_1 , act_type='relu')
    conv3_2 = mx.symbol.Convolution(name='conv3_2', data=relu3_1 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_2 = mx.symbol.Activation(name='relu3_2', data=conv3_2 , act_type='relu')
    conv3_3 = mx.symbol.Convolution(name='conv3_3', data=relu3_2 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_3 = mx.symbol.Activation(name='relu3_3', data=conv3_3 , act_type='relu')
    conv3_4 = mx.symbol.Convolution(name='conv3_4', data=relu3_3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu3_4 = mx.symbol.Activation(name='relu3_4', data=conv3_4 , act_type='relu')
    pool3 = mx.symbol.Pooling(name='pool3', data=relu3_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv4_1 = mx.symbol.Convolution(name='conv4_1', data=pool3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_1 = mx.symbol.Activation(name='relu4_1', data=conv4_1 , act_type='relu')
    conv4_2 = mx.symbol.Convolution(name='conv4_2', data=relu4_1 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_2 = mx.symbol.Activation(name='relu4_2', data=conv4_2 , act_type='relu')
    conv4_3 = mx.symbol.Convolution(name='conv4_3', data=relu4_2 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_3 = mx.symbol.Activation(name='relu4_3', data=conv4_3 , act_type='relu')
    conv4_4 = mx.symbol.Convolution(name='conv4_4', data=relu4_3 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu4_4 = mx.symbol.Activation(name='relu4_4', data=conv4_4 , act_type='relu')
    pool4 = mx.symbol.Pooling(name='pool4', data=relu4_4 , pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
    conv5_1 = mx.symbol.Convolution(name='conv5_1', data=pool4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False, workspace=1024)
    relu5_1 = mx.symbol.Activation(name='relu5_1', data=conv5_1 , act_type='relu')

    print(style_layers)
    print(content_layers)
    out = mx.sym.Group([x for x in map(eval, style_layers)])
    out = mx.sym.Group([out] + [x for x in map(eval, content_layers)])
    return out


