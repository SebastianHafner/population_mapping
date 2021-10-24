import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from utils import paths, experiment_manager
from copy import deepcopy
from collections import OrderedDict
from sys import stderr
import copy


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
        pt = cfg.MODEL.PRETRAINED
        if cfg.MODEL.TYPE == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=pt)
        elif cfg.MODEL.TYPE == 'alexnet':
            self.model = torchvision.models.alexnet(pretrained=pt)
        elif cfg.MODEL.TYPE == 'vgg16':
            self.model = torchvision.models.vgg16(pretrained=pt)
        elif cfg.MODEL.TYPE == 'squeezenet1':
            self.model = torchvision.models.squeezenet1_0(pretrained=pt)
        elif cfg.MODEL.TYPE == 'densenet161':
            self.model = torchvision.models.densenet161(pretrained=pt)
        elif cfg.MODEL.TYPE == 'inceptionv3':
            self.model = torchvision.models.inception_v3(pretrained=pt)
        elif cfg.MODEL.TYPE == 'googlenet':
            self.model = torchvision.models.googlenet(pretrained=pt)
        elif cfg.MODEL.TYPE == 'shufflenet':
            self.model = torchvision.models.shufflenet_v2_x1_0(pretrained=pt)
        elif cfg.MODEL.TYPE == 'mobilnetv2':
            self.model = torchvision.models.mobilenet_v2(pretrained=pt)
        elif cfg.MODEL.TYPE == 'resnext5032x4d':
            self.model = torchvision.models.resnext50_32x4d(pretrained=pt)
        elif cfg.MODEL.TYPE == 'wideresnet502':
            self.model = torchvision.models.wide_resnet50_2(pretrained=pt)
        elif cfg.MODEL.TYPE == 'mnasnet':
            self.model = torchvision.models.mnasnet1_0(pretrained=pt)
        else:
            self.model = None

        new_in_channels = len(cfg.DATALOADER.SATELLITE_BANDS)
        if new_in_channels != 3:
            conv1_layer = self.model.conv1

            # Creating new Conv2d layer
            new_conv1_layer = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=conv1_layer.out_channels,
                kernel_size=conv1_layer.kernel_size,
                stride=conv1_layer.stride,
                padding=conv1_layer.padding,
                bias=conv1_layer.bias
            )
            # he initialization
            nn.init.kaiming_uniform_(new_conv1_layer.weight.data, mode='fan_in', nonlinearity='relu')

            # replace weights of first 3 channels with resnet rgb ones
            conv1_layer_weights = conv1_layer.weight.data.clone()
            new_conv1_layer.weight.data[:, :conv1_layer.in_channels, :, :] = conv1_layer_weights
            self.model.conv1 = new_conv1_layer
            # https://discuss.pytorch.org/t/how-to-change-no-of-input-channels-to-a-pretrained-model/19379/2
            # https://discuss.pytorch.org/t/how-to-modify-the-input-channels-of-a-resnet-model/2623/10
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, cfg.MODEL.OUT_CHANNELS)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.relu(x)
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