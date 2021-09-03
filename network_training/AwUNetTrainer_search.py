import torch
import torch.nn.functional as F
import numpy as np
from nnunet.network_architecture.aw_UNet_search import AwUNet_search
from nnunet.training.network_training.AdaptiveUNetTrainer_search import AdaptiveUNetTrainer_search
from nnunet.utilities.nd_softmax import softmax_helper

class AwUNetTrainer_search(AdaptiveUNetTrainer_search):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        

        self.conv_num_per_stage = 2

    def initialize_network(self):
        
        self.network = AwUNet_search(self.num_input_channels, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_num_per_stage,
                                    self.net_num_pool_op_kernel_sizes,
                                    self.net_conv_kernel_sizes)

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        self.num_stages = len(self.net_num_pool_op_kernel_sizes) + 1

    def print_current_arch(self):
        arch_code = []    
        self.print_to_log_file('Current architecture parameters:')
        for param in self.network.arch_parameters():
            param = F.softmax(param.detach().cpu(), dim=-1)
            param = param.numpy()
            arch_code.append(np.argmax(param))
            self.print_to_log_file(' '.join(['{:.6f}'.format(p) for p in param]))

        assert len(arch_code) == (self.num_stages * 2 - 1) * self.conv_num_per_stage

        stage = 0
        current_channels = []
        for i in range(self.num_stages * 2 - 1 ):

            stage = i if i < self.num_stages else self.num_stages * 2 - 2 - i
            stage = min(stage, 3)
            channel_gap = 8 * 2 ** stage
            min_channel = channel_gap * 2

            current_channels.append([])
            for j in range(self.conv_num_per_stage):
                current_channels[-1].append(min_channel+channel_gap*arch_code[i*self.conv_num_per_stage+j])
                
        self.print_to_log_file('Current arch:', current_channels)