from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.adaptive_UNet_blocks import ConvDropoutNormNonlin, StackedConvLayers2


class AdwUNet(SegmentationNetwork):

    def __init__(self, input_channels, num_classes, num_pool, arch_list,
                 pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None):
        assert len(arch_list) == num_pool * 2 + 1
        super(AdwUNet, self).__init__()

        self.input_channels = input_channels
        self.num_classes = self.num_classes
        self.num_pool = num_pool
        self.arch_list = arch_list

        basic_block = ConvDropoutNormNonlin

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

        input_features = input_channels

        for d in range(num_pool+1):
            if d != 0:
                first_stride = pool_op_kernel_sizes[d-1]
            else:
                first_stride = None
            
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]

            # StackedConvLayers2   arch_list[d]
            self.conv_blocks_context.append(StackedConvLayers2(input_features, arch_list[d],
                                                                      self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs,
                                                                      self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                                      first_stride, basic_block=basic_block))
            
            input_features = arch_list[d][-1]
        
        final_num_features = input_features

        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            

            self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 2)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 2)]

            self.conv_blocks_localization.append(StackedConvLayers2(n_features_after_tu_and_concat, arch_list[num_pool+1+u], 
                                                                           self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs,
                                                                           self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                                           first_stride=None, basic_block=basic_block))

            final_num_features = self.conv_blocks_localization[-1].output_channels

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(self.conv_op(self.conv_blocks_localization[ds].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

        # register all modules properly
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        
        self.apply(self.weightInitializer)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
