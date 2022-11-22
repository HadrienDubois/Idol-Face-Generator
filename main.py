#To run on google colab, uncomment the part below

#!nvidia-smi -L                  #check the GPU available on the execution environment
#from google.colab import drive  
#drive.mount('/content/drive')   #connect to your personnal drive

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
else:
    !git clone https://github.com/dvschultz/stylegan2-ada-pytorch
    %cd stylegan2-ada-pytorch
    !mkdir downloads
    !mkdir datasets
    !mkdir pretrained
    %cd pretrained
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

#update the last_model with the last pkl file to generate faces
last_model = '/content/drive/MyDrive/colab-sg2-ada-pytorch/stylegan2-ada-pytorch/results/00012-Final_idols_dataset-mirror-11gb-gpu-gamma50-kimg1000-bg-resumecustom/network-snapshot-000320.pkl'

#update the outdir variable to select the directory where the generated faces will be saved
!python generate.py --outdir=/content/out/images/ --trunc=0.8 --seeds=0 --network=$last_model
