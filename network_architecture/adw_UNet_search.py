# Adaptive depth and width U-Net for search
# 可以同时调整网络深度和宽度的网络结构

from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.adaptive_UNet_blocks import AdaptiveWidthConvDropoutNormNonlin, AdaptiveDepthWidthStackedConvLayers


class AdwUNet_search(SegmentationNetwork):

    def __init__(self, input_channels, num_classes, max_num_convs,
                 num_pool,
                 pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None):

        super(AdwUNet_search, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes


        base_num_features = 32
        feat_map_mul_on_downscale = 2
        basic_block=AdaptiveWidthConvDropoutNormNonlin

        self.conv_op = nn.Conv3d
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        transpconv = nn.ConvTranspose3d

        self.nonlin = nn.LeakyReLU
        self.nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

        self.dropout_op = nn.Dropout3d
        self.dropout_op_kwargs = {'p': 0, 'inplace': True}

        self.norm_op = nn.InstanceNorm3d
        self.norm_op_kwargs = {'eps': 1e-5, 'affine': True}

        self.weightInitializer = InitWeights_He(1e-2)
        self.num_classes = num_classes
        self.final_nonlin = lambda x:x  
        self._deep_supervision = True
        self.do_ds = True
        seg_output_use_bias=False
        self.upscale_logits = False

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])


        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.tu = []
        self.seg_outputs = []

        output_features = int(base_num_features*1.5)
        input_features = input_channels
        self.max_num_features = 384
        self.max_num_convs = max_num_convs

        for d in range(num_pool+1):
            # determine the first stride
            if d != 0:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(AdaptiveDepthWidthStackedConvLayers(input_features, output_features, self.max_num_convs,
                                                                           self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs,
                                                                           self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                                           first_stride, basic_block=basic_block))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)

        final_num_features = input_features

        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            final_num_features = nfeatures_from_skip

            self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                       pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 2)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 2)]

            self.conv_blocks_localization.append(AdaptiveDepthWidthStackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, self.max_num_convs,
                                                 self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs,
                                                 self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                 basic_block=basic_block))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(self.conv_op(self.conv_blocks_localization[ds].output_channels, num_classes,
                                                 1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)
        
        # register all modules properly
        # 修改了原始的数据，先注册下采样路，再注册上采样路
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

        self.apply(self.weightInitializer)

    def forward(self, x, tau):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x, tau)
            skips.append(x)

        x = self.conv_blocks_context[-1](x, tau)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x, tau)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    def arch_parameters(self):
        _arch_parameters = []
        for k, v in self.named_parameters():
            if k.endswith('alphas') or k.endswith('betas'):
                _arch_parameters.append(v)
        return _arch_parameters

    def weight_parameters(self):
        _weight_parameters = []
        for k, v in self.named_parameters():
            if not (k.endswith('alphas') or k.endswith('betas')):
                _weight_parameters.append(v)
        return _weight_parameters

    
    def VRAM_estimate(self, patch_size):
        # 为简便，先仅计算feature map的占用，忽略参数占用

        depth_params = []
        width_params = []
        for param in self.arch_parameters():
            if len(param) == self.max_num_convs:
                depth_params.append(F.softmax(param, dim=-1))
            else:
                width_params.append(F.softmax(param, dim=-1))
        
        # 先将宽度结构变量转成等效宽度
        stage = 0
        equal_arch_list = []
        for i in range(len(depth_params)):
            stage = i if i <= len(depth_params) // 2 else len(depth_params) - i - 1
            stage = min(stage, 3)
            channel_gap = 8 * 2 ** stage
            min_channel = channel_gap * 2

            equal_channels = []
            for j in range(self.max_num_convs):
                equal_channels.append(0)
                for k in range(5):
                    equal_channels[-1] += width_params[i*self.max_num_convs+j][k] * (min_channel + k * channel_gap)
            equal_arch_list.append(equal_channels)

        # 计算每个深度结构变量对应的显存
        map_size = patch_size.copy()
        tmp = 0
        pool_op_kernel_sizes = np.array(self.pool_op_kernel_sizes)
        tmp += self.input_channels * np.prod(map_size)

        for i in range(len(self.pool_op_kernel_sizes)):
            for j in range(self.max_num_convs):
                for k in range(j+1):
                    tmp += equal_arch_list[i][k] * np.prod(map_size) * depth_params[i][j]
                    tmp += equal_arch_list[-i-1][k] * np.prod(map_size) * depth_params[-i-1][j]
                tmp += equal_arch_list[i][j] * np.prod(map_size) *depth_params[i][j]

            map_size //= pool_op_kernel_sizes[i]

        for j in range(self.max_num_convs):
            for k in range(j+1):
                tmp += equal_arch_list[len(self.pool_op_kernel_sizes)][k] * np.prod(map_size) * depth_params[len(self.pool_op_kernel_sizes)][j]

        return tmp*2/256/1024
