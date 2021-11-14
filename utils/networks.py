import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from utils import paths, experiment_manager
from copy import deepcopy
from collections import OrderedDict
from sys import stderr


def save_checkpoint(network, optimizer, epoch, step, cfg: experiment_manager.CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = {
        'step': step,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(epoch, cfg: experiment_manager.CfgNode, device):
    net = PopulationNet(cfg)
    net.to(device)

    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}_checkpoint{epoch}.pt'
    checkpoint = torch.load(save_file, map_location=device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['step']


def create_ema_network(net, cfg):
    ema_net = EMA(net, decay=cfg.CONSISTENCY_TRAINER.WEIGHT_DECAY)
    return ema_net


class PopulationNet(nn.Module):

    def __init__(self, cfg):
        super(PopulationNet, self).__init__()
        self.cfg = cfg
        pt = cfg.MODEL.PRETRAINED
        if cfg.MODEL.TYPE == 'resnet':
            if cfg.MODEL.SIZE == 18:
                self.model = torchvision.models.resnet18(pretrained=pt)
            elif cfg.MODEL.SIZE == 50:
                self.model = torchvision.models.resnet50(pretrained=pt)
            else:
                raise Exception(f'Unkown resnet size ({cfg.MODEL.SIZE}).')
        elif cfg.MODEL.TYPE == 'densenet':
            if cfg.MODEL.SIZE == 121:
                self.model = torchvision.models.densenet121(pretrained=pt)
            elif cfg.MODEL.SIZE == 161:
                self.model = torchvision.models.densenet161(pretrained=pt)
            else:
                raise Exception(f'Unkown densenet size ({cfg.MODEL.SIZE}).')
        else:
            if cfg.MODEL.TYPE == 'alexnet':
                self.model = torchvision.models.alexnet(pretrained=pt)
            elif cfg.MODEL.TYPE == 'vgg16':
                self.model = torchvision.models.vgg16(pretrained=pt)
            elif cfg.MODEL.TYPE == 'squeezenet1':
                self.model = torchvision.models.squeezenet1_0(pretrained=pt)
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
            elif cfg.MODEL.TYPE == 'efficientnetb4':
                self.model = torchvision.models.efficientnet_b4(pretrained=pt)
            elif cfg.MODEL.TYPE == 'wideresnet50':
                self.model = torchvision.models.wide_resnet50_2(pretrained=pt)
            else:
                raise Exception(f'Unkown network ({cfg.MODEL.TYPE}).')

        new_in_channels = cfg.MODEL.IN_CHANNELS
        if new_in_channels != 3:
            # only implemented for resnet and densenet
            assert(cfg.MODEL.TYPE == 'resnet' or cfg.MODEL.TYPE == 'densenet')

            # if cfg.MODEL.TYPE == 'resnet':
            first_layer = self.model.conv1 if cfg.MODEL.TYPE == 'resnet' else self.model.features.conv0
            # Creating new Conv2d layer
            new_first_layer = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=first_layer.out_channels,
                kernel_size=first_layer.kernel_size,
                stride=first_layer.stride,
                padding=first_layer.padding,
                bias=first_layer.bias
            )
            # he initialization
            nn.init.kaiming_uniform_(new_first_layer.weight.data, mode='fan_in', nonlinearity='relu')
            if new_in_channels > 3:

                # replace weights of first 3 channels with resnet rgb ones
                first_layer_weights = first_layer.weight.data.clone()
                new_first_layer.weight.data[:, :first_layer.in_channels, :, :] = first_layer_weights

            if cfg.MODEL.TYPE == 'resnet':
                # replacing first layer
                self.model.conv1 = new_first_layer
                # replacing fully connected layer
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, cfg.MODEL.OUT_CHANNELS)
                # https://discuss.pytorch.org/t/how-to-change-no-of-input-channels-to-a-pretrained-model/19379/2
                # https://discuss.pytorch.org/t/how-to-modify-the-input-channels-of-a-resnet-model/2623/10
            else:
                # replacing first layer
                self.model.features.conv0 = new_first_layer
                # adding fully connected layer
                self.fc = nn.Linear(1_000, cfg.MODEL.OUT_CHANNELS)

        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        if self.cfg.MODEL.TYPE == 'densenet':
            x = self.fc(x)
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