import torch
from nnunet.network_architecture.adw_UNet import AdwUNet
from nnunet.training.network_training.AdaptiveUNetTrainer import AdaptiveUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper


class AdwUNetTrainer(AdaptiveUNetTrainer):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        # 设置结构参数
        task_name = plans_file.split('/')[-2]    
        self.arch_list = None
        
        if task_name == 'Task001_BrainTumour':
            self.arch_list = [[48, 48, 48], [96, 96], [192, 192], [384, 384], [320, 384, 320], [384, 384, 384], [384, 384, 384], [384, 384, 384], [192, 192], [96, 96], [40, 40]] 
            
        elif task_name == 'Task002_Heart':   
            self.arch_list = [[48, 48], [80, 80, 96], [192, 192], [384], [256, 256, 320], [384, 128, 320], [256, 384], [384, 320], [160, 192], [80, 80], [24]]
            
        elif task_name == 'Task005_Prostate':
            self.arch_list = [[48, 32, 32], [80, 96], [192, 192, 192], [384, 256], [384, 320, 384], [320, 384], [384, 256, 384], [384, 384, 384], [256, 320, 384], [384, 320, 320], [192], [48], [48, 48, 16]]

        elif task_name == 'Task007_Pancreas':
            self.arch_list = [[48, 48, 48], [96, 96, 96], [192, 192, 192], [384, 384, 384], [320, 256, 320], [384, 320, 384], [384, 384], [320, 320, 384], [192, 192, 192], [96, 96, 80], [48, 32, 40]]

        elif task_name == "Task817_FLARE":
            self.arch_list = [[48, 48], [96, 96], [192, 192, 192], [384, 384, 384], [384, 384, 384], [192, 320, 128], [384, 320, 320], [384, 384, 384], [192, 192, 192], [96], [48]] 
            

    def initialize_network(self):

        self.network = AdwUNet(self.num_input_channels, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.arch_list,
                                    self.net_num_pool_op_kernel_sizes, 
                                    self.net_conv_kernel_sizes)
        print('current arch:', self.arch_list)
        print('VRAM USAGE:', self.network.compute_approx_vram_consumption(self.patch_size))
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
