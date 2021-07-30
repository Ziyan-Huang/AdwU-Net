import torch
from torch import nn
from copy import deepcopy
import torch.nn.functional as F

# Basic Blocks

# 结构固定的基本卷积单元
class ConvDropoutNormNonlin(nn.Module):

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


# 用于搜索的通道数可变的卷积单元
class AdaptiveWidthConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, max_output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(AdaptiveWidthConvDropoutNormNonlin, self).__init__()

        self.max_output_channels = max_output_channels

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op_kwargs = dropout_op_kwargs
        self.dropout_op = dropout_op
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op

        self.conv = self.conv_op(input_channels, self.max_output_channels, **self.conv_kwargs)

        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(self.max_output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

        channel_gap = int(max_output_channels / 6)
        min_channel_num = int(max_output_channels / 3)
        self.channel_map = list(range(min_channel_num, max_output_channels+1, channel_gap))
        assert  len(self.channel_map) == 5

        self._initialize_alphas()
        self._initialize_masks()
        
    def forward(self, x, tau):
        x = self.conv(x)
        weights = F.gumbel_softmax(self.alphas, tau=tau, dim=-1)
        weighted_mask = sum(w * mask for w, mask in zip(weights, self.masks))
        expand_mask = weighted_mask.view(-1,1,1,1)
        x = x * expand_mask
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))

    def _initialize_alphas(self):
        alphas = torch.zeros((5))
        self.register_parameter('alphas', nn.Parameter(alphas))

    def _initialize_masks(self):
        # 数据类型是否要用bool？
        masks = torch.zeros((5, self.max_output_channels))
        for index, n_channel in enumerate(self.channel_map):
            masks[index, :n_channel] = 1
        self.register_buffer('masks', nn.Parameter(masks, requires_grad=False))


# Stacked Blocks 

# 结构固定的连续卷积，num_convs代表卷积的次数，每次卷积的输出通道数都为output_feature_channels
# 用于AdUNet重训练
class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


# 结构固定的连续卷积，output_feature_channels是一个列表，依次代表每个卷积的输出通道数
# 用于AwUNet重训练
class StackedConvLayers2(nn.Module):
    # output_feature_channels是一个列表
    def __init__(self, input_feature_channels, output_feature_channels,
                 conv_op, conv_kwargs, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                 first_stride=None, basic_block=ConvDropoutNormNonlin):
        super(StackedConvLayers2, self).__init__()

        self.num_convs = len(output_feature_channels)
        assert self.num_convs in [1,2,3,4]
        # 记录下该StackedLayers最后一层的output channels，在大网络中会用到
        self.output_channels = output_feature_channels[-1]

        if first_stride is not None:
            conv_kwargs_first_conv = deepcopy(conv_kwargs)
            conv_kwargs_first_conv['stride'] = first_stride
        else:
            conv_kwargs_first_conv = conv_kwargs

        self.block1 = basic_block(input_feature_channels, output_feature_channels[0],
                                conv_op, conv_kwargs_first_conv, norm_op, norm_op_kwargs,
                                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        if self.num_convs >=2 :
            self.block2 = basic_block(output_feature_channels[0], output_feature_channels[1],
                                    conv_op, conv_kwargs, norm_op, norm_op_kwargs,
                                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        if self.num_convs >=3 :
            self.block3 = basic_block(output_feature_channels[1], output_feature_channels[2],
                                    conv_op, conv_kwargs, norm_op, norm_op_kwargs,
                                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        if self.num_convs >=4 :
            self.block4 = basic_block(output_feature_channels[2], output_feature_channels[3],
                                    conv_op, conv_kwargs, norm_op, norm_op_kwargs,
                                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
    def forward(self, x):
        res_list = []
        out1 = self.block1(x)
        res_list.append(out1)

        if self.num_convs >= 2:
            out2 = self.block2(out1)
            res_list.append(out2)
        if self.num_convs >= 3:
            out3 = self.block3(out2)
            res_list.append(out3)
        if self.num_convs >= 4:
            out4 = self.block4(out3)
            res_list.append(out4)
        
        return res_list[-1]



# 用于搜索的深度可变的若干次卷积
# 用于AdUNet的搜索
class AdaptiveDepthStackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, max_num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):

        super(AdaptiveDepthStackedConvLayers, self).__init__()
        
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels


        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.max_num_convs = max_num_convs

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs
       
        # 注意第一个block的输入输出通道不同，还有conv_kwargs不同，还有first_stride
        self.block1 = basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs_first_conv,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs) 
        if max_num_convs >= 2:
            self.block2 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)
        if max_num_convs >= 3:
            self.block3 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)
        if max_num_convs >= 4:                
            self.block4 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)
        self._initialize_betas()

    def forward(self, x, tau):
        res_list = []
        out1 = self.block1(x)
        res_list.append(out1)
        if self.max_num_convs >= 2:
            out2 = self.block2(out1)
            res_list.append(out2)
        if self.max_num_convs >= 3:
            out3 = self.block3(out2)
            res_list.append(out3)
        if self.max_num_convs >= 4:
            out4 = self.block4(out3)
            res_list.append(out4)

        weights = F.gumbel_softmax(self.betas, tau=tau, dim=-1)
        assert len(weights) == len(res_list)
        out = sum(w*res for w, res in zip(weights, res_list))
        return out
    
    def _initialize_betas(self):
        # 定义结构参数betas
        assert self.max_num_convs in [2,3,4]
        betas = torch.zeros((self.max_num_convs))
        self.register_parameter('betas', nn.Parameter(betas))


# 用于搜索的宽度可变若干次卷积，深度固定
# 用于AwUNet的搜索
class AdaptiveWidthStackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=AdaptiveWidthConvDropoutNormNonlin):
        super(AdaptiveWidthStackedConvLayers, self).__init__()
        
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.num_convs = num_convs      

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        
        assert num_convs in [1,2,3,4]
       
        # 注意第一个block的输入输出通道不同，还有conv_kwargs不同
        self.block1 = basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs_first_conv,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)

        if num_convs >= 2: 
            self.block2 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)
        if num_convs >= 3:
            self.block3 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)
        if num_convs >= 4:                
            self.block4 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)

    def forward(self, x, tau):
        res_list = []
        out1 = self.block1(x, tau)
        res_list.append(out1)

        if self.num_convs >= 2:
            out2 = self.block2(out1, tau)
            res_list.append(out2)
        if self.num_convs >= 3:
            out3 = self.block3(out2, tau)
            res_list.append(out3)
        if self.num_convs >= 4:
            out4 = self.block4(out3, tau)
            res_list.append(out4)
        
        return res_list[-1]



# 用于AdwUNet的搜索

class AdaptiveDepthWidthStackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, max_num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=AdaptiveWidthConvDropoutNormNonlin):

        super(AdaptiveDepthWidthStackedConvLayers, self).__init__()
        
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels


        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.max_num_convs = max_num_convs

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs
       
        # 注意第一个block的输入输出通道不同，还有conv_kwargs不同，还有first_stride
        self.block1 = basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs_first_conv,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs) 
        if max_num_convs >= 2:
            self.block2 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)
        if max_num_convs >= 3:
            self.block3 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)
        if max_num_convs >= 4:                
            self.block4 = basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                            self.conv_kwargs,
                            self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                            self.nonlin, self.nonlin_kwargs)
        self._initialize_betas()

    def forward(self, x, tau):
        res_list = []
        out1 = self.block1(x, tau)
        res_list.append(out1)
        if self.max_num_convs >= 2:
            out2 = self.block2(out1, tau)
            res_list.append(out2)
        if self.max_num_convs >= 3:
            out3 = self.block3(out2, tau)
            res_list.append(out3)
        if self.max_num_convs >= 4:
            out4 = self.block4(out3, tau)
            res_list.append(out4)

        weights = F.gumbel_softmax(self.betas, tau=tau, dim=-1)
        assert len(weights) == len(res_list)
        out = sum(w*res for w, res in zip(weights, res_list))
        return out
    
    def _initialize_betas(self):
        # 定义结构参数betas
        assert self.max_num_convs in [2,3,4]
        betas = torch.zeros((self.max_num_convs))
        self.register_parameter('betas', nn.Parameter(betas))


