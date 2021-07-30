import torch
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch.nn.functional as F
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.network_architecture.neural_network import SegmentationNetwork

from collections import OrderedDict

import numpy as np
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from _warnings import warn
from torch.optim.lr_scheduler import _LRScheduler
from time import time
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold

from torch.utils.tensorboard import SummaryWriter
import datetime


class AdaptiveUNetTrainer_search(nnUNetTrainerV2):

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 250
        self.begin_search_epoch = 100
        self.initial_lr_w = 1e-2
        self.initial_lr_a = 4e-4
        self.trA_percent = 0.5
        self.weight_decay_a = 0
        self.initial_tau = 5
        self.tau_anneal_rate = 0.97
        self.tau = self.initial_tau

        # 增加结构one-hot限制loss
        self.lamb = 0 
        self.arch_constraint_loss = ArchConstraintLoss(10000, 0.5)

        dt = datetime.datetime.now().strftime("%m-%d-%I-%M%p")
        self.writer = SummaryWriter(join(self.output_folder, "logs", dt))

        # 搜索程序跑的慢，保存checkpoint频率高一点
        self.save_every = 20


    def initialize(self, training=True, force_load_plans=False):
        
        # 打印本次程序使用的超参数
        self.print_to_log_file("max_num_epochs: %d" % self.max_num_epochs)
        self.print_to_log_file("begin_search_epoch: %d" % self.begin_search_epoch)
        self.print_to_log_file("trA_percent: %.2f" % self.trA_percent)
        self.print_to_log_file("weight_decay_a: %f" % self.weight_decay_a)
        self.print_to_log_file("initial tau: %d" % self.initial_tau)
        self.print_to_log_file("tau_anneal_rate: %.2f" % self.tau_anneal_rate)
        self.print_to_log_file("arch_loss_lambda: %e" % self.lamb)

        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_trA, self.dl_trB, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                # 将这一段重复两遍即可？
                self.trA_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_trA, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.trB_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_trB, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING A KEYS:\n %s" % (str(self.dataset_trA.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("TRAINING B KEYS:\n %s" % (str(self.dataset_trB.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        # 每次继承需要自己写
        pass

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer_w = torch.optim.SGD(self.network.weight_parameters(), self.initial_lr_w, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.optimizer_a = torch.optim.Adam(self.network.arch_parameters(), self.initial_lr_a, weight_decay=self.weight_decay_a)
        self.optimizer = self.optimizer_w
        # 没有用到
        self.lr_scheduler = None


    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        dl_trA = DataLoader3D(self.dataset_trA, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        dl_trB = DataLoader3D(self.dataset_trB, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 False, oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides, memmap_mode='r')
        
        return dl_trA, dl_trB, dl_val

    def do_split(self):
       
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            splits = load_pickle(splits_file)

            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
            else:
                self.print_to_log_file("INFO: Requested fold %d but split file only has %d folds. I am now creating a "
                                       "random 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]

        tr_keys.sort()
        val_keys.sort()
        # 开始修改
        self.dataset_trA = OrderedDict()
        self.dataset_trB = OrderedDict()
        self.dataset_val = OrderedDict()

        rnd = np.random.RandomState(seed=2021)
        idx_trA = rnd.choice(len(tr_keys), int(len(tr_keys) * self.trA_percent), replace=False)
        idx_trB = [i for i in range(len(tr_keys)) if i not in idx_trA]
        trA_keys = [tr_keys[i] for i in idx_trA]
        trB_keys = [tr_keys[i] for i in idx_trB]
        for i in trA_keys:
            self.dataset_trA[i] = self.dataset[i]
        for i in trB_keys:
            self.dataset_trB[i] = self.dataset[i]  
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]


    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        ds = self.network.do_ds
        self.network.do_ds = True
        
        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['plans']
        del dct['intensity_properties']
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_trA']
        del dct['dataset_trB']
        del dct['dataset_val']
        
        save_json(dct, join(self.output_folder, "debug.json"))

        import shutil

        shutil.copy(self.plans_file, join(self.output_folder_base, "plans.pkl"))

    
        _ = self.trA_gen.next()
        _ = self.trB_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)
        
        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            weight_losses_epoch = []

            # train one epoch
            self.network.train()

            for _ in range(self.num_batches_per_epoch):
                # MiLeNAS
                # 训练结构参数，使用全部训练数据
                if self.epoch >= self.begin_search_epoch:
                    for param in self.network.weight_parameters():
                        param.requires_grad = False
                    for param in self.network.arch_parameters():
                        param.requires_grad = True
                    _ = self.run_iteration(self.trA_gen, self.optimizer_a, True)
                    _ = self.run_iteration(self.trB_gen, self.optimizer_a, True)

                # 训练模型参数，使用trainB
                for param in self.network.weight_parameters():
                    param.requires_grad = True
                for param in self.network.arch_parameters():
                    param.requires_grad = False
                loss_w = self.run_iteration(self.trB_gen, self.optimizer_w, True)
                weight_losses_epoch.append(loss_w)

            # 打印结构并且存储在tensorboard中，各搜索函数中独立实现
            if self.epoch >= self.begin_search_epoch:
                self.print_current_arch()

            self.all_tr_losses.append(np.mean(weight_losses_epoch))
            self.print_to_log_file("Train weight loss: %.4f" % self.all_tr_losses[-1])
            # 存在Tensorboard中
            self.writer.add_scalar('Loss/train', self.all_tr_losses[-1], self.epoch)
            
            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, self.optimizer_w, False, True)   # 此处optimizer没有用到
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("Validation loss: %.4f" % self.all_val_losses[-1])
                # 记录在Tensorboard
                self.writer.add_scalar('Loss/val', self.all_val_losses[-1], self.epoch)
                # 去掉also_val_in_train的选择

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        # if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # 注释掉搜索阶段最终模型的保存

        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

        self.network.do_ds = ds
    
    def print_current_arch(self):
        arch_code = []  
        stage = 0  
        # 打印并获取结构参数
        self.print_to_log_file('Current architecture parameters:')
        for param in self.network.arch_parameters():
            stage = stage + 1
            param = F.softmax(param.detach().cpu(), dim=-1)
            param = param.numpy()
            # self.writer.add_histogram('Arch Distribution %d'%(stage), param, self.epoch)
            arch_code.append(np.argmax(param)+1)
            self.print_to_log_file(' '.join(['{:.6f}'.format(p) for p in param]))
        self.print_to_log_file('Current arch:', arch_code) 

        

    def run_iteration(self, data_generator, optimizer, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        optimizer.zero_grad()

        if self.fp16:
            with autocast():
                # 前向传播
                output, l = self.forward_out_and_loss(data, target)
                del data

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(optimizer)
                self.amp_grad_scaler.update()
        else:
            output, l = self.forward_out_and_loss(data, target)
            del data

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def forward_out_and_loss(self, data, target):
        output = self.network(data, self.tau)
        l = self.loss(output, target) # + self.lamb * (self.epoch/self.max_num_epochs) * self.arch_constraint_loss(self.network.VRAM_estimate(self.patch_size))
        return output, l


    # 将原始一个optimizer的学习率改为两个optimizer的学习率变动，加上tau的衰减
    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        
        self.optimizer_w.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr_w, 0.9)
        self.print_to_log_file("lr_a:", np.round(self.optimizer_a.param_groups[0]['lr'], decimals=6))
        self.print_to_log_file("lr_w:", np.round(self.optimizer_w.param_groups[0]['lr'], decimals=6))

        if ep > self.begin_search_epoch:
            self.tau = self.tau_anneal_rate * self.tau
        self.print_to_log_file("tau", np.round(self.tau, decimals=6))


    # 模型的保存和加载，对于optimizer都要改成optimizer_a和optimizer_w
    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            
        if save_optimizer:
            optimizer_w_state_dict = self.optimizer_w.state_dict()
            optimizer_a_state_dict = self.optimizer_a.state_dict()
        else:
            optimizer_w_state_dict = None
            optimizer_a_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_w_state_dict': optimizer_w_state_dict,
            'optimizer_a_state_dict': optimizer_a_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
            'tau': self.tau,
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_w_state_dict = checkpoint['optimizer_w_state_dict']
            optimizer_a_state_dict = checkpoint['optimizer_a_state_dict']
            if optimizer_w_state_dict is not None:
                self.optimizer_w.load_state_dict(optimizer_w_state_dict)
            if optimizer_a_state_dict is not None:
                self.optimizer_a.load_state_dict(optimizer_a_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']
        self.tau = checkpoint['tau']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()

    def manage_patience(self):
        # update patience
        continue_training = True
        
        # 搜索程序不需要保存checkpoints
        # if self.best_val_eval_criterion_MA is None:
        #     self.best_val_eval_criterion_MA = self.val_eval_criterion_MA 

        # if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
        #     self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
        #     self.print_to_log_file("saving best epoch checkpoint...")
        #     self.save_checkpoint(join(self.output_folder, "model_best.model"))

        #     if not self.save_latest_only:
        #         self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d_best.model" % (self.epoch + 1)))

        return continue_training


class ArchConstraintLoss(nn.Module):
    # 将每组结构参数与它相应的one-hot形式，进行L1 Norm
    def __init__(self, target_vram, sigma):
        super(ArchConstraintLoss, self).__init__()
        self.target_vram = target_vram
        self.sigma = sigma
    def forward(self, estimate_vram):
        return torch.abs(estimate_vram / self.target_vram - self.sigma)
