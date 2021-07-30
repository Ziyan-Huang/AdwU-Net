import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from nnunet.training.network_training.AdaptiveUNetTrainer_search import AdaptiveUNetTrainer_search
from nnunet.network_architecture.adw_UNet_search import AdwUNet_search
from nnunet.utilities.nd_softmax import softmax_helper

import matplotlib.pyplot as plt

class AdwUNetTrainer_search(AdaptiveUNetTrainer_search):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 250
        self.begin_search_epoch = 100
        self.initial_tau = 5
        self.tau_anneal_rate = 0.97
        self.tau = self.initial_tau
        self.max_num_convs = 3
        self.initial_lr_a = 4e-4
        self.trA_percent = 0.5



    def initialize_network(self):
       
        self.network = AdwUNet_search(self.num_input_channels, self.num_classes, self.max_num_convs,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.net_num_pool_op_kernel_sizes,
                                    self.net_conv_kernel_sizes)

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


    def print_current_arch(self):
        depth_code = []    
        width_code = []
        depth_parameters_softmax = []
        width_parameters_softmax = []
        # 打印并获取结构参数
        self.print_to_log_file('Current architecture parameters:')
        for param in self.network.arch_parameters():
            param = F.softmax(param.detach().cpu(), dim=-1)
            # 判断是不是深度结构参数
            if len(param) == self.max_num_convs:                
                depth_parameters_softmax.append(param)
                param = param.numpy()
                depth_code.append(np.argmax(param)+1)
            else:
                width_parameters_softmax.append(param)
                param = param.numpy()
                width_code.append(np.argmax(param))
            self.print_to_log_file(' '.join(['{:.6f}'.format(p) for p in param]))

        assert len(width_code) == self.max_num_convs * len(depth_code)

        current_channels = []
        stage = 0
        for i in range(len(depth_code)):

            stage = i if i <= len(depth_code) // 2 else len(depth_code) - i - 1
            stage = min(stage, 3)
            channel_gap = 8 * 2 ** stage
            min_channel = channel_gap * 2

            current_channels.append([])
            for j in range(depth_code[i]):
                current_channels[-1].append(min_channel + channel_gap * width_code[i*self.max_num_convs+j])
        self.print_to_log_file('Current arch:', current_channels)

        
        # 保存heatmap到tensorboard中
        width_weight_array = torch.stack(width_parameters_softmax)
        width_weight_array = width_weight_array.detach().cpu()
        depth_weight_array = torch.stack(depth_parameters_softmax)
        depth_weight_array = depth_weight_array.detach().cpu()

        fig = plt.figure(figsize=(10,5))
        ax_depth = fig.add_subplot(1,2,2)
        ax_depth.set_yticks(np.arange(len(depth_weight_array)))
        ax_depth.set_xticks(np.arange(len(depth_weight_array[0])))
        ax_depth.set_yticklabels(depth_code)
        im = ax_depth.imshow(depth_weight_array, cmap=plt.cm.Greens, vmin=0, vmax=1)
        ax_depth.figure.colorbar(im, ax=ax_depth)

        ax_width = fig.add_subplot(1,2,1)
        ax_width.set_yticks(np.arange(0, len(width_weight_array), self.max_num_convs))
        ax_width.set_xticks(np.arange(len(width_weight_array[0])))
        ax_width.set_yticklabels(current_channels)
        im = ax_width.imshow(width_weight_array, cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax_width.figure.colorbar(im, ax=ax_width)

        self.writer.add_figure('arch weight', fig, self.epoch)
