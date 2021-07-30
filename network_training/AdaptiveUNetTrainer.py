# 让所有重训练程序从该文件继承，避免对nnUNetTrainerV2直接修改

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from torch.utils.tensorboard import SummaryWriter

class AdaptiveUNetTrainer(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        # 为了加快调参效率，减少一些总训练轮数
        self.max_num_epochs = 600
        self.initial_lr = 1e-2
        # 改变数值，原始默认值0.9
        self.val_eval_criterion_alpha = 0.8

    def manage_patience(self):

        if self.best_val_eval_criterion_MA is None:
            self.best_val_eval_criterion_MA = self.val_eval_criterion_MA 

        if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
            self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
            self.print_to_log_file("saving best epoch checkpoint...")
            self.save_checkpoint(join(self.output_folder, "model_best.model"))

            if self.epoch>100:
                self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d_best.model" % (self.epoch + 1)))
        continue_training = True
        return continue_training

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)
        # 更改，num_thread以及cached，原参数为12和2
        # self.data_aug_params["num_threads"] = 12
        # self.data_aug_params["num_cached_per_thread"] = 2
        self.data_aug_params["num_threads"] = 2
        self.data_aug_params["num_cached_per_thread"] = 1

