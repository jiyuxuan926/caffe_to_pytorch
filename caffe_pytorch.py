from caffenet import *

def load_image(imgfile):
    import caffe
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, 224, 224)})
    transformer.set_transpose('data', (2, 0, 1))
    #transformer.set_mean('data', np.array([args.meanB, args.meanG, args.meanR]))
    #transformer.set_raw_scale('data', args.scale)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, 224, 224)
    return image

def forward_pytorch(protofile, weightfile, image):
    net = CaffeNet(protofile)
    print(net)
    net.load_weights(weightfile)
    net.eval()
    image = torch.from_numpy(image)
    image = Variable(image)
    blobs = net(image)
    return blobs, net.models

imgfile = 'data/cat.jpg'
protofile = 'model/ResNet-50-deploy.prototxt'
weightfile = 'model/ResNet-50-model.caffemodel'
image = load_image(imgfile)
pytorch_blobs, pytorch_models = forward_pytorch(protofile, weightfile, image)
torch.save(pytorch_models,'a.pth')
model_a = torch.load('a.pth')
print model_a
print pytorch_blobs