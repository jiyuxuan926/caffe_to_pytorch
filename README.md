forked from marvis/pytorch-caffe, and details of usage on https://github.com/marvis/pytorch-caffe,
This repository does some modifications on supported layers.
## caffe2pytorch
This tool aims to load caffe prototxt and weights directly in pytorch without explicitly converting model from caffe to pytorch.

### Usage
use the model and weight directly in pytorch, and save the model as.pth file

Download [caffemodel of resnet50](https://pan.baidu.com/s/1YnvN8Hy9cZLqFtKPqs4X-g), unzip it to model.
Then
```
python caffe_pytorch.py
```
### Supported Layers
We modify the convolution and deconvolution layer, so their kernel size can be int of kernel_size or tuple of(kernel_h, kernel_w)
Each layer in caffe will have a corresponding layer in pytorch. 
- [x] Convolution
- [x] InnerProduct
- [x] BatchNorm
- [x] Scale
- [x] ReLU
- [x] Pooling
- [x] Reshape
- [x] Softmax
- [x] Accuracy
- [x] SoftmaxWithLoss
- [x] Dropout
- [x] Eltwise
- [x] Normalize
- [x] Permute
- [x] Flatten
- [x] Slice
- [x] Concat
- [x] PriorBox
- [x] LRN : gpu version is ok, cpu version produce big difference
- [x] DetectionOutput: support batchsize=1, num_classes=1 forward
- [x] Crop
- [x] Deconvolution
- [x] MultiBoxLoss
