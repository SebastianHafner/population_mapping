import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from utils import paths, experiment_manager


def load_network(cfg: experiment_manager.CfgNode, checkpoint: int = None):
    net = CustomNet(cfg)
    dirs = paths.load_paths()
    checkpoint = checkpoint if checkpoint is None else cfg.INFERENCE.CHECKPOINT
    net_file = Path(dirs.OUTPUT) / f'{cfg.NAME}_{checkpoint}.pkl'
    state_dict = torch.load(str(net_file), map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict)
    return net


def save_checkpoint(network, optimizer, epoch, step, cfg):
    dirs = paths.load_paths()
    save_file = Path(dirs.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = {
        'step': step,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(epoch, cfg, device):
    net = CustomNet(cfg)
    net.to(device)

    dirs = paths.load_paths()
    save_file = Path(dirs.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = torch.load(save_file, map_location=device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['step']


class CustomNet(nn.Module):

    def __init__(self, cfg):
        super(CustomNet, self).__init__()
        self.cfg = cfg

        if cfg.MODEL.TYPE == 'resnet18':
            model = torchvision.models.resnet18()
        elif cfg.MODEL.TYPE == 'alexnet':
            model = torchvision.models.alexnet()
        elif cfg.MODEL.TYPE == 'vgg16':
            model = torchvision.models.vgg16()
        elif cfg.MODEL.TYPE == 'squeezenet1':
            model = torchvision.models.squeezenet1_0()
        elif cfg.MODEL.TYPE == 'densenet161':
            model = torchvision.models.densenet161()
        elif cfg.MODEL.TYPE == 'inceptionv3':
            model = torchvision.models.inception_v3()
        elif cfg.MODEL.TYPE == 'googlenet':
            model = torchvision.models.googlenet()
        else:
            model = None
        # shufflenet = models.shufflenet_v2_x1_0()
        # mobilenet_v2 = models.mobilenet_v2()
        # mobilenet_v3_large = models.mobilenet_v3_large()
        # mobilenet_v3_small = models.mobilenet_v3_small()
        # resnext50_32x4d = models.resnext50_32x4d()
        # wide_resnet50_2 = models.wide_resnet50_2()
        # mnasnet = models.mnasnet1_0()

        # changing the input channels of the first layer
        in_channels = cfg.IN_CHANNELS
        first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        first_conv_layer.extend(list(model.features))
        model.features = nn.Sequential(*first_conv_layer)

        # Add a avgpool here
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Replace the classifier layer
        self.vgg11.classifier[-1] = nn.Linear(4096, cfg.OUT_CHANNELS)


    def forward(self, x):
        x = self.vgg11.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.vgg11.classifier(x)
        return x
