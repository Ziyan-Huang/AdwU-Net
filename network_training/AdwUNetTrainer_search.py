import torch
import torch.nn.functional as F
import numpy as np
from nnunet.training.network_training.AdaptiveUNetTrainer_search import AdaptiveUNetTrainer_search
from nnunet.network_architecture.adw_UNet_search import AdwUNet_search
from nnunet.utilities.nd_softmax import softmax_helper

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
        self.print_to_log_file('Current architecture parameters:')
        for param in self.network.arch_parameters():
            param = F.softmax(param.detach().cpu(), dim=-1)

            if len(param) == self.max_num_convs:                
                param = param.numpy()
                depth_code.append(np.argmax(param)+1)
            else:
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

