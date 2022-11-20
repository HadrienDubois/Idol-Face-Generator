!nvidia-smi -L #check the GPU available on google colab

from google.colab import drive
drive.mount('/content/drive')

import os
!pip install gdown --upgrade

if os.path.isdir("/content/drive/MyDrive/colab-sg2-ada-pytorch"):
    %cd "/content/drive/MyDrive/colab-sg2-ada-pytorch/stylegan2-ada-pytorch"
elif os.path.isdir("/content/drive/"):
    #install script
    %cd "/content/drive/MyDrive/"
    !mkdir colab-sg2-ada-pytorch
    %cd colab-sg2-ada-pytorch
    !git clone https://github.com/dvschultz/stylegan2-ada-pytorch
    %cd stylegan2-ada-pytorch
    !mkdir downloads
    !mkdir datasets
    !mkdir pretrained
    !gdown --id 1-5xZkD8ajXw1DdopTkH_rAoCsD72LhKU -O /content/drive/MyDrive/colab-sg2-ada-pytorch/stylegan2-ada-pytorch/pretrained/wikiart.pkl
else:
    !git clone https://github.com/dvschultz/stylegan2-ada-pytorch
    %cd stylegan2-ada-pytorch
    !mkdir downloads
    !mkdir datasets
    !mkdir pretrained
    %cd pretrained
    !gdown --id 1-5xZkD8ajXw1DdopTkH_rAoCsD72LhKU
    %cd ../
    
#Uninstall new JAX
!pip uninstall jax jaxlib -y
#GPU frontend
!pip install "jax[cuda11_cudnn805]==0.3.10" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#CPU frontend
#!pip install jax[cpu]==0.3.10
#Downgrade Pytorch
!pip uninstall torch torchvision -y
!pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
!pip install timm==0.4.12 ftfy==6.1.1 ninja==1.10.2 opensimplex


#Parameters to edit in order to train the model
dataset_path = '/content/drive/MyDrive/Final_idols_dataset.zip'
resume_from = '/content/drive/MyDrive/ffhq-512-avg-tpurun1.pkl'
aug_strength = 0.0
train_count = 0
mirror_x = True
#mirror_y = False

#optional parameters
gamma_value = 25.0
augs = 'bg'
config = '11gb-gpu'
snapshot_count = 4
total_kimg = 1000


#Edit the resume_from with the last pkl file created to train the model
!python train.py --gpus=1 --cfg=$config --metrics=None --outdir=./results --data=$dataset_path --snap=$snapshot_count --resume=$resume_from --augpipe=$augs --initstrength=$aug_strength --gamma=$gamma_value --mirror=$mirror_x --kimg=$total_kimg --mirrory=False --nkimg=$train_count

#update the network with the last pkl file to generate faces
last_model = '/content/drive/MyDrive/colab-sg2-ada-pytorch/stylegan2-ada-pytorch/results/00012-Final_idols_dataset-mirror-11gb-gpu-gamma50-kimg1000-bg-resumecustom/network-snapshot-000320.pkl'

!python generate.py --outdir=/content/out/images/ --trunc=0.8 --seeds=0 --network=$last_model
