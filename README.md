# AdwU-Net: Adaptive depth and Width U-Net
Built upon [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), this repository provides the official PyTorch implementation of [AdwU-Net: Adaptive depth and width U-Net for medical image segmentation](https://openreview.net/forum?id=kF-d1SKWJpS).

## Environments and Requirements:
1.Install nnU-Net [1] as below. You should meet the requirements of nnUNet, our method does not need any additional requirements.  For more details, please refer to https://github.com/MIC-DKFZ/nnUNet
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

2.Set environment variables for nnU-Net. Concretely, Set the paths in your .bashrc file, which is located in your home directory. Open the file and add the following lines to the bottom:
```
export nnUNet_raw_data_base="/data/hzy/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/data/hzy/nnUNet/nnUNet_preprocessed"
export RESULTS_FOLDER="/data/hzy/nnUNet/nnUNet_trained_models"
```
(of course adapt the paths to your system)

3.Copy the python files in this repository to the code directory of nnUNet.
```
cp network_training/* nnunet/training/network_training/
cp network_architecture/* nnunet/network_architecture/
cp run_searching.py nnunet/run/
```

## How to use AdwU-Net
For example, we search the optimal depth and width of Task005_Prostate and then configure the optimal depth and width of nnUNet to train the model.
* Run the following command to search the optimal depth and width on Task005_Prostate
```
cd nnunet/run
python run_searching.py 3d_fullres AdwUNetTrainer_search 5 all
```
* Edit  AdwUNetTrainer.py, change the value of self.arch_list using the output at the end of search stage.
* Run the following command to train nnUNet using the optimal depth and width on Task005_Prostate
```
for FOLD in 0 1 2 3 4
do
nnUNet_train 3d_fullres AdwUNetTrainer 5 $FOLD
done
```
## Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{
huang2022adwunet,
title={AdwU-Net: Adaptive Depth and Width U-Net for Medical Image Segmentation by Differentiable Neural Architecture Search},
author={Ziyan Huang and Zehua Wang and zhikai yang and Lixu Gu},
booktitle={Medical Imaging with Deep Learning},
year={2022},
url={https://openreview.net/forum?id=kF-d1SKWJpS}
}
```
