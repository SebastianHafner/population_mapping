import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from utils import paths, datasets, experiment_manager, networks, evaluation, geofiles
from pathlib import Path
import math
FONTSIZE = 16


def qualitative_assessment_celllevel(config_name: str, run_type: str = 'test', n_samples: int = 30, scale_factor: float = 0.3):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.PopulationMappingDataset(cfg, run_type, no_augmentations=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    plot_size = 3
    n_cols = 5
    n_rows = n_samples // n_cols
    if n_samples % n_cols != 0:
        n_rows += 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*plot_size, n_rows*plot_size))

    indices = np.random.randint(0, len(ds), n_samples)
    for index, item_index in enumerate(tqdm(indices)):
        item = ds.__getitem__(item_index)
        x = item['x']
        pred_pop = net(x.to(device).unsqueeze(0)).flatten().cpu().item()
        pop = item['y'].item()

        img = x.cpu().numpy().transpose((1, 2, 0))
        img = np.clip(img / scale_factor, 0, 1)

        i = index // n_cols
        j = index % n_cols
        ax = axs[i, j] if n_rows > 1 else axs[index]
        ax.imshow(img)
        ax.set_title(f'Pred: {pred_pop: .0f} - Pop: {pop:.0f}')
        ax.set_axis_off()
    plt.show()
    plt.close(fig)


def visualize_outliers(config_name: str, run_type: str = 'test'):
    pass


def visualize_high_pop(config_name: str, run_type: str = 'test', pop_min: int = 500, n_samples: int = 5):
    pass


def correlation_celllevel(config_name: str, run_type: str = 'test', add_lin_regression: bool = True,
                          add_1to1: bool = True, add_density: bool = True, scale: str = 'linear'):
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


def quantitative_assessment_celllevel(config_name: str, run_type: str = 'test'):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.PopulationMappingDataset(cfg, run_type, no_augmentations=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    measurer = evaluation.RegressionEvaluation()

    for i, index in enumerate(tqdm(range(len(ds)))):
        item = ds.__getitem__(index)
        x = item['x']
        pred_pop = net(x.to(device).unsqueeze(0)).flatten().cpu()
        pop = item['y'].cpu()
        measurer.add_sample(pred_pop, pop)

        if i == 100:
            pass
    rmse = measurer.root_mean_square_error()
    print(f'RMSE: {rmse:.2f}')


def run_quantitative_assessment_censuslevel(config_name: str, city: str, run_type: str = 'test'):
    cfg = experiment_manager.load_cfg(config_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()


    dirs = paths.load_paths()
    census_data = geofiles.load_json(Path(dirs.DATASET) / 'census_data' / f'{run_type}_polygons_dakar.geojson')
    for f in census_data['features']:
        pop_census = 0
        pred_pop_census = 0
        poly_id = f['properties']['poly_id']
        ds = datasets.CensusDataset(cfg, city, run_type, int(poly_id))
        if ds.valid_for_assessment:
            for i, index in enumerate(tqdm(range(len(ds)))):
                item = ds.__getitem__(index)
                x = item['x']
                pred_pop = net(x.to(device).unsqueeze(0)).flatten().cpu()
                pop = item['y'].cpu()
                pop_census += pop.item()
                pred_pop_census += pred_pop.item()

            pop_census_ref = f['properties']['POPULATION']
            f['properties']['population_pred'] = pred_pop_census
            print(f'Census Pop: {pop_census_ref}; Pred: {pred_pop_census}; Sum pop grid: {pop_census}')
        else:
            f['properties']['population_pred'] = np.NaN

    out_file = Path(dirs.OUTPUT) / 'predictions' / f'{config_name}_{run_type}_{city}.geojson'
    geofiles.write_json(out_file, census_data)


def correlation_censuslevel(config_name: str, city: str, run_type: str = 'test', scale: str = 'linear'):
    dirs = paths.load_paths()
    pred_file = Path(dirs.OUTPUT) / 'predictions' / f'{config_name}_{run_type}_{city}.geojson'
    if not pred_file.exists():
        run_quantitative_assessment_censuslevel(config_name, city, run_type)

    data = geofiles.load_json(pred_file)
    gts = [float(f['properties']['POPULATION']) for f in data['features']]
    preds = [float(f['properties']['population_pred']) for f in data['features']]
    nans = np.isnan(preds)
    gts = np.array(gts)[~nans]
    preds = np.array(preds)[~nans]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(gts, preds, c='k', s=10)

    slope, intercept, r_value, p_value, std_err = stats.linregress(gts, preds)
    x = np.array([0, 1_000])
    ax.plot(x, slope * x + intercept, c='k')
    # place a text box in upper left in axes coords
    textstr = r'$R^2 = {r_value:.2f}$'.format(r_value=r_value)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=FONTSIZE,
            verticalalignment='top')

    pop_max = 100_000
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


if __name__ == '__main__':
    config = 'resnet18_baseline'
    # qualitative_assessment_celllevel(config)
    # run_quantitative_assessment_censuslevel(config, 'dakar')
    correlation_censuslevel(config, 'dakar')
    # correlation(config, add_lin_regression=True, scale='linear')
    # quantitative_assessment(config)