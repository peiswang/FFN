# FFN
Code for papers:
* 'Fixed-Point Factorized Networks', CVPR 2017
* 'Unsupervised Network Quantization via Fixed-point Factorization', TNNLS

Fixed-point Factorized Network (FFN) is a novel network ternarization approach, i.e., it turns all weights into ternary values {-1, 0, 1}. FFN works well in both training-aware and post-training quantization schemes. It can achieve negligible degradation even without any supervised finetuning on labeled data.

Factorized VGG16 model:
[BaiduCloud](https://pan.baidu.com/s/1RUyS1rVAuvDYyzM-UK1bOw)

# Train:
    python main_ffn_vgg.py --pretrained <path-to-pretrained-model>
    
# Test:
    python main_ffn_vgg_test.py --pretrained vgg16_bn/vgg16_ternary_final.pth


# Results:

 \* Acc@1 70.810 Acc@5 90.050

# Related Papers

Please cite our paper if it helps your research:

    @InProceedings{Wang_2017_CVPR,
      author = {Wang, Peisong and Cheng, Jian},
      title = {Fixed-Point Factorized Networks},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {July},
      year = {2017}
    } 
