# 让所有重训练程序从该文件继承，避免对nnUNetTrainerV2直接修改

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from batchgenerators.utilities.file_and_folder_operations import join

class AdaptiveUNetTrainer(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.val_eval_criterion_alpha = 0.9

    def manage_patience(self):

        if self.best_val_eval_criterion_MA is None:
            self.best_val_eval_criterion_MA = self.val_eval_criterion_MA 

        if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
            self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
            self.print_to_log_file("saving best epoch checkpoint...")
            self.save_checkpoint(join(self.output_folder, "model_best.model"))
            
        continue_training = True
        return continue_training
