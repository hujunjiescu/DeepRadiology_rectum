deeplab_resnet50 = {
    'os' : 16,
    'backbone' : 'ResNet50',
    'pretrain_checkpoint': '/root/.torch/models/resnet50-19c8e357.pth',
    'ignore_prefixs': ["conv1", "fc", "gap"]
}