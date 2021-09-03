from nnunet.network_architecture.aw_UNet import AwUNet
import torch
from nnunet.training.network_training.AdaptiveUNetTrainer import AdaptiveUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper


class AwUNetTrainer(AdaptiveUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        task_name = plans_file.split('/')[-2]
        
            
        if task_name == 'Task005_Prostate':
            
            self.channels_per_stages = [[32, 40], [96, 96], [160, 160], [384, 256], [384, 192], [192, 384], [384, 384], [384, 384], [384, 384], [320, 384], [192, 64], [96, 32], [48, 24]]

        

    def initialize_network(self):

        self.network = AwUNet(self.num_input_channels, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.channels_per_stages,
                                    self.net_num_pool_op_kernel_sizes, 
                                    self.net_conv_kernel_sizes)
        
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper