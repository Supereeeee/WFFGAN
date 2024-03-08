# Wavelet-based feature fusion generative adversarial network for single image super-resolution(WFFGAN)
Yinggan Tang, Quanwei Hu, Chunning BU

## Environment in our experiments
[python 3.8]

[Ubuntu 18.04]

[BasicSR 1.4.2](https://github.com/XPixelGroup/BasicSR)

[PyTorch 1.13.0, Torchvision 0.14.0, Cuda 11.7](https://pytorch.org/get-started/previous-versions/)

### Installation
```
git clone https://github.com/Supereeeee/WFFGAN.git
pip install -r requirements.txt
python setup.py develop
```

## How To Test
· Refer to ./options/test for the configuration file of the model to be tested and prepare the testing data.  

· The pre-trained models have been palced in ./experiments/pretrained_models/  

· You can test the results of PSNR-oriented WFFGAN by running the follwing codes:  
```
python basicsr/test.py -opt options/test/test_WFFGAN_PSNR_x4.yml
```
· You can test the results of perception-oriented WFFGAN by running the follwing codes:  
```
python basicsr/test.py -opt options/test/test_WFFGAN_x4.yml
```
All testing results will be saved in the ./results folder.

## How To Train
· Refer to ./options/train for the configuration file of the model to train.  

· Preparation of training data can refer to this page. All datasets can be downloaded at the official website.  

· Note that the default training dataset is based on lmdb, refer to [docs in BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) to learn how to generate the training datasets.  

· We divide the training process into two stages: A PSNR-oriented generator for pre-training and a perception-oriented WFFGAN using pre-trained generator

· The training command for PSNR-oriented generator is like:  
```
python basicsr/train.py -opt options/train/train_WFFGAN_PSNR_x4.yml
```
· The training command for perception-oriented WFFGAN using pre-trained generator is like:  
```
python basicsr/train.py -opt options/train/train_WFFGAN_x4.yml
```
For more training commands and details, please check the docs in [BasicSR](https://github.com/XPixelGroup/BasicSR)  


## Inference
· You can run ./inference/inference_WFFGAN.py for your own images.


## Acknowledgement
This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. Thanks for the awesome work.

## Contact
If you have any question, please email 1051823707@qq.com.
