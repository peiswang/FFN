# FFN
Code for paper 'Unsupervised Network Quantization via Fixed-point Factorization'

Factorized VGG16 model:
[BaiduCloud](https://pan.baidu.com/s/1RUyS1rVAuvDYyzM-UK1bOw)

Usage:

    python main_ffn_vgg.py --pretrained <path-to-pretrained-model>

    python main_ffn_vgg_test.py --pretrained vgg16_bn/vgg16_ternary_final.pth


Results:

 \* Acc@1 70.810 Acc@5 90.050

'''
@InProceedings{Wang_2017_CVPR,
author = {Wang, Peisong and Cheng, Jian},
title = {Fixed-Point Factorized Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
} 
'''
