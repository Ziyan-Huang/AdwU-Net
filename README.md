# AdwU-Net: Adaptive depth and Width U-Net

Built upon [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), this repository provides the official PyTorch implementation of AdwU-Net.

## How to use AdwU-Net:
### 1. Requirements:
Linux, Python3.7+, Pytorch1.6+
### 2. Installation:
* Install nnU-Net as below
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
* Copy the python files in folder network_architecture to nnunet/architecture
* Copy the python files in folder network_training to nnunet/training/network_training
* Copy the python file run_searching.py to nnunet/run
### 3. Running the scripts:
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

