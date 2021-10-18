import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from utils import paths, experiment_manager
from copy import deepcopy
from collections import OrderedDict
from sys import stderr


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


def create_ema_network(net, cfg):
    ema_net = EMA(net, decay=cfg.CONSISTENCY_TRAINER.WEIGHT_DECAY)
    return ema_net


class CustomNet(nn.Module):

    def __init__(self, cfg):
        super(CustomNet, self).__init__()
        self.cfg = cfg

        if cfg.MODEL.TYPE == 'resnet18':
            self.model = torchvision.models.resnet18()
        elif cfg.MODEL.TYPE == 'alexnet':
            self.model = torchvision.models.alexnet()
        elif cfg.MODEL.TYPE == 'vgg16':
            self.model = torchvision.models.vgg16()
        elif cfg.MODEL.TYPE == 'squeezenet1':
            self.model = torchvision.models.squeezenet1_0()
        elif cfg.MODEL.TYPE == 'densenet161':
            self.model = torchvision.models.densenet161()
        elif cfg.MODEL.TYPE == 'inceptionv3':
            self.model = torchvision.models.inception_v3()
        elif cfg.MODEL.TYPE == 'googlenet':
            self.model = torchvision.models.googlenet()
        else:
            self.model = None
        # shufflenet = models.shufflenet_v2_x1_0()
        # mobilenet_v2 = models.mobilenet_v2()
        # mobilenet_v3_large = models.mobilenet_v3_large()
        # mobilenet_v3_small = models.mobilenet_v3_small()
        # resnext50_32x4d = models.resnext50_32x4d()
        # wide_resnet50_2 = models.wide_resnet50_2()
        # mnasnet = models.mnasnet1_0()

        # changing the input channels of the first layer
        first_conv_layer = [nn.Conv2d(cfg.MODEL.IN_CHANNELS, 3, kernel_size=3, stride=1, padding=1, dilation=1,
                                      groups=1, bias=True)]
        first_conv_layer.extend(list(self.model.features))
        self.model.features = nn.Sequential(*first_conv_layer)

        # Add a avgpool here
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Replace the classifier layer
        self.model.classifier[-1] = nn.Linear(4096, cfg.MODEL.OUT_CHANNELS)


    def forward(self, x):
        x = self.model.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.model.classifier(x)
        return x


# https://www.zijianhu.com/post/pytorch/ema/
class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.ema_model = deepcopy(self.model)

        for param in self.ema_model.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        ema_model_params = OrderedDict(self.ema_model.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == ema_model_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            ema_model_params[name].sub_((1. - self.decay) * (ema_model_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        ema_model_buffers = OrderedDict(self.ema_model.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == ema_model_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            ema_model_buffers[name].copy_(buffer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.ema_model(inputs)

    def get_ema_model(self):
        return self.ema_model

    def get_model(self):
        return self.model



if __name__ == '__main__':
    x = torch.randn(1, 5, 224, 224)
    model = torchvision.models.vgg16(pretrained=False)  # pretrained=False just for debug reasons
    first_conv_layer = [nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(list(model.features))
    model.features = nn.Sequential(*first_conv_layer)
    output = model(x)