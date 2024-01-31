import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from utils import experiment_manager
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
    net = DualStreamPopulationNet(cfg.MODEL) if cfg.MODEL.DUALSTREAM else PopulationNet(cfg.MODEL)
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


class DualStreamPopulationNet(nn.Module):

    def __init__(self, dual_model_cfg: experiment_manager.CfgNode):
        super(DualStreamPopulationNet, self).__init__()
        self.dual_model_cfg = dual_model_cfg
        self.stream1_cfg = dual_model_cfg.STREAM1
        self.stream2_cfg = dual_model_cfg.STREAM2

        self.stream1 = PopulationNet(self.stream1_cfg, enable_fc=False)
        self.stream2 = PopulationNet(self.stream2_cfg, enable_fc=False)

        stream1_num_ftrs = self.stream1.model.fc.in_features
        stream2_num_ftrs = self.stream2.model.fc.in_features
        self.outc = nn.Linear(stream1_num_ftrs + stream2_num_ftrs, dual_model_cfg.OUT_CHANNELS)
        self.relu = torch.nn.ReLU()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        features1 = self.stream1(x1)
        features2 = self.stream2(x2)
        p1 = self.relu(self.stream1.model.fc(features1))
        p2 = self.relu(self.stream2.model.fc(features2))
        features_fusion = torch.cat((features1, features2), dim=1)
        p_fusion = self.relu(self.outc(features_fusion))
        return p_fusion, p1, p2


class PopulationNet(nn.Module):

    def __init__(self, model_cfg, enable_fc: bool = True):
        super(PopulationNet, self).__init__()
        self.model_cfg = model_cfg
        self.enable_fc = enable_fc
        pt = model_cfg.PRETRAINED
        assert (model_cfg.TYPE == 'resnet')
        if model_cfg.SIZE == 18:
            self.model = torchvision.models.resnet18(pretrained=pt)
        elif model_cfg.SIZE == 50:
            self.model = torchvision.models.resnet50(pretrained=pt)
        else:
            raise Exception(f'Unkown resnet size ({model_cfg.SIZE}).')

        new_in_channels = model_cfg.IN_CHANNELS

        if new_in_channels != 3:
            # only implemented for resnet
            assert (model_cfg.TYPE == 'resnet')

            first_layer = self.model.conv1
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
            # if it is less than 3 channels we use he initialization (no pretraining)

            # replacing first layer
            self.model.conv1 = new_first_layer
            # https://discuss.pytorch.org/t/how-to-change-no-of-input-channels-to-a-pretrained-model/19379/2
            # https://discuss.pytorch.org/t/how-to-modify-the-input-channels-of-a-resnet-model/2623/10

        # replacing fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, model_cfg.OUT_CHANNELS)
        self.relu = torch.nn.ReLU()
        self.encoder = torch.nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_fc:
            x = self.model(x)
            x = self.relu(x)
        else:
            x = self.encoder(x)
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        return x


class PopulationNetMultiTask(nn.Module):

    def __init__(self, model_cfg):
        super(PopulationNetMultiTask, self).__init__()
        self.model_cfg = model_cfg
        pt = model_cfg.PRETRAINED
        assert (model_cfg.TYPE == 'resnet')
        if model_cfg.SIZE == 18:
            self.model = torchvision.models.resnet18(pretrained=pt)
        elif model_cfg.SIZE == 50:
            self.model = torchvision.models.resnet50(pretrained=pt)
        else:
            raise Exception(f'Unkown resnet size ({model_cfg.SIZE}).')

        new_in_channels = model_cfg.IN_CHANNELS

        if new_in_channels != 3:
            # only implemented for resnet
            assert (model_cfg.TYPE == 'resnet')

            first_layer = self.model.conv1
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
            # if it is less than 3 channels we use he initialization (no pretraining)

            # replacing first layer
            self.model.conv1 = new_first_layer
            # https://discuss.pytorch.org/t/how-to-change-no-of-input-channels-to-a-pretrained-model/19379/2
            # https://discuss.pytorch.org/t/how-to-modify-the-input-channels-of-a-resnet-model/2623/10

        # replacing fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, model_cfg.OUT_CHANNELS)
        self.relu = torch.nn.ReLU()
        self.encoder = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.fc_class = nn.Linear(num_ftrs, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        f = torch.flatten(f, start_dim=1)
        out1 = self.relu(self.model.fc(f))
        out2 = self.fc_class(f)

        if self.training:
            return out1, out2
        else:
            out2 = torch.argmax(self.softmax(out2), dim=1)
            out = out1 * torch.logical_not(out2.bool()).float()
            return out