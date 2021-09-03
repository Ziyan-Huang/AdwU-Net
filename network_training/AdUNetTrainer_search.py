import torch
import torch.nn.functional as F
from nnunet.training.network_training.AdaptiveUNetTrainer_search import AdaptiveUNetTrainer_search
from nnunet.network_architecture.ad_UNet_search import AdUNet_search
from nnunet.utilities.nd_softmax import softmax_helper
import numpy as np

class AdUNetTrainer_search(AdaptiveUNetTrainer_search):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        self.max_num_convs_per_stage = 3


    def initialize_network(self):
       
        self.network = AdUNet_search(self.num_input_channels, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.max_num_convs_per_stage,
                                    self.net_num_pool_op_kernel_sizes,
                                    self.net_conv_kernel_sizes)

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def print_current_arch(self):
        arch_code = []  
        arch_parameters_softmax = []
 
        # 打印并获取结构参数
        self.print_to_log_file('Current architecture parameters:')
        for param in self.network.arch_parameters():
            param = F.softmax(param.detach().cpu(), dim=-1)
            arch_parameters_softmax.append(param)
            param = param.numpy()
            # self.writer.add_histogram('Arch Distribution %d'%(stage), param, self.epoch)
            arch_code.append(np.argmax(param)+1)
            self.print_to_log_file(' '.join(['{:.6f}'.format(p) for p in param]))
        self.print_to_log_file('Current arch:', arch_code) 
