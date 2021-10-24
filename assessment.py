import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from utils import datasets, experiment_manager, networks, metrics
FONTSIZE = 16


def qualitative_assessment(config_name: str, run_type: str = 'test', n_samples: int = 5, scale_factor: float = 0.3):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.PopulationMappingDataset(cfg, run_type, no_augmentations=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    indices = np.random.randint(0, len(ds), n_samples)
    for index in indices:
        item = ds.__getitem__(index)
        x = item['x']
        pred_pop = net(x.to(device).unsqueeze(0)).flatten().cpu().item()
        pop = item['y'].item()

        fig, ax = plt.subplots(1, 1)
        img = x.cpu().numpy().transpose((1, 2, 0))
        img = np.clip(img / scale_factor, 0, 1)
        ax.imshow(img)
        ax.set_title(f'Pop: {pop:.0f}, Pred: {pred_pop:.0f}')
        ax.set_axis_off()
        plt.show()
        plt.close(fig)


def visualize_outliers(config_name: str, run_type: str = 'test'):
    pass


def visualize_high_pop(config_name: str, run_type: str = 'test', pop_min: int = 500, n_samples: int = 5):
    pass


def correlation(config_name: str, run_type: str = 'test', add_lin_regression: bool = True, add_1to1: bool = True,
                add_density: bool = True, scale: str = 'linear'):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.PopulationMappingDataset(cfg, run_type, no_augmentations=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    preds, gts = [], []
    for i, index in enumerate(tqdm(range(len(ds)))):
        item = ds.__getitem__(index)
        x = item['x']
        pred_pop = net(x.to(device).unsqueeze(0)).flatten().cpu().item()
        preds.append(pred_pop)
        pop = item['y'].item()
        gts.append(pop)
        if i == 100:
            pass

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    if add_density:
        # Calculate the point density
        xy = np.vstack([gts, preds])
        z = stats.gaussian_kde(xy)(xy)
        ax.scatter(gts, preds, c=z, s=10)
    else:
        ax.scatter(gts, preds, c='k')
    if add_lin_regression:
        slope, intercept, r_value, p_value, std_err = stats.linregress(gts, preds)
        x = np.array([0, 1_000])
        ax.plot(x, slope * x + intercept, c='k')
        # place a text box in upper left in axes coords
        textstr = r'$R^2 = {r_value:.2f}$'.format(r_value=r_value)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=FONTSIZE,
                verticalalignment='top')

    pop_max = 1_000
    if scale == 'linear':
        ticks = np.linspace(0, pop_max, 5)
        pop_min = 0
    else:
        ticks = [1, 10, 100, 1_000]
        ax.set_xscale('log')
        ax.set_yscale('log')
        pop_min = 1
    ax.set_xlim(pop_min, pop_max)
    ax.set_ylim(pop_min, pop_max)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'{tick:.0f}' for tick in ticks], fontsize=FONTSIZE)
    ax.set_yticklabels([f'{tick:.0f}' for tick in ticks], fontsize=FONTSIZE)
    ax.set_xlabel('Ground Truth', fontsize=FONTSIZE)
    ax.set_ylabel('Prediction', fontsize=FONTSIZE)

    plt.show()


def quantitative_assessment(config_name: str, run_type: str = 'test'):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.PopulationMappingDataset(cfg, run_type, no_augmentations=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    measurer = metrics.

    preds, gts = [], []
    for i, index in enumerate(tqdm(range(len(ds)))):
        item = ds.__getitem__(index)
        x = item['x']
        pred_pop = net(x.to(device).unsqueeze(0)).flatten().cpu().item()
        preds.append(pred_pop)
        pop = item['y'].item()
        gts.append(pop)
        if i == 100:
            pass
    pass


if __name__ == '__main__':
    config = 'resnet18_baseline'
    # qualitative_assessment(config)
    correlation(config, add_lin_regression=True, scale='linear')
