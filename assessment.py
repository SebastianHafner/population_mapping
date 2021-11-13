import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines
from tqdm import tqdm
from scipy import stats
from utils import paths, datasets, experiment_manager, networks, evaluation, geofiles
from pathlib import Path
import math
FONTSIZE = 16
# TODO: add support for pop log


def qualitative_assessment_celllevel(config_name: str, run_type: str = 'test', n_samples: int = 30, scale_factor: float = 0.3):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.CellPopulationDataset(cfg, run_type, no_augmentations=True)
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
    dirs = paths.load_paths()
    out_file = Path(dirs.OUTPUT) / 'plots' / f'dakar_qualitative_assessment_{config_name}.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def correlation_celllevel(config_name: str, city: str, run_type: str = 'test', scale: str = 'linear'):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.CellPopulationDataset(cfg, run_type, no_augmentations=True)
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
    # Calculate the point density
    xy = np.vstack([gts, preds])
    z = stats.gaussian_kde(xy)(xy)
    markersize = 10
    ax.scatter(gts, preds, c=z, s=markersize, label='Cell')

    slope, intercept, r_value, p_value, std_err = stats.linregress(gts, preds)
    x = np.array([0, 1_000])
    # ax.plot(x, slope * x + intercept, c='k')
    # place a text box in upper left in axes coords
    textstr = r'$R^2 = {r_value:.2f}$'.format(r_value=r_value)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=FONTSIZE,
            verticalalignment='top')

    pop_max = 1_000
    line = ax.plot([0, pop_max], [0, pop_max], c='k', zorder=-1, label='1:1 line')
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
    legend_elements = [
        lines.Line2D([0], [0], color='k', lw=1, label='1:1 Line'),
        lines.Line2D([0], [0], marker='.', color='w', markerfacecolor='k', label='Cell', markersize=markersize),
    ]
    ax.legend(handles=legend_elements, fontsize=FONTSIZE, frameon=False, loc='upper center')
    dirs = paths.load_paths()
    out_file = Path(dirs.OUTPUT) / 'plots' / f'{city}_correlation_celllevel_{config_name}.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.show()


def quantitative_assessment_celllevel(config_name: str, run_type: str = 'test'):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.CellPopulationDataset(cfg, run_type, no_augmentations=True)
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
        pop_census, pred_pop_census = 0, 0
        poly_id = f['properties']['poly_id']
        ds = datasets.CensusPopulationDataset(cfg, city, run_type, int(poly_id))
        if ds.valid_for_assessment and ds.length > 0:
            f['properties']['valid'] = True
            for i, index in enumerate(tqdm(range(len(ds)))):
                item = ds.__getitem__(index)
                x = item['x']
                pred_pop = net(x.to(device).unsqueeze(0)).flatten().cpu()
                pop = item['y'].cpu()
                pop_census += pop.item()
                pred_pop_census += pred_pop.item()

            pop_census_ref = f['properties']['POPULATION']
            f['properties']['population_pred'] = pred_pop_census
            print(f'Id: {poly_id}: Census Pop: {pop_census_ref} - Pred: {pred_pop_census}')
        else:
            f['properties']['population_pred'] = np.NaN
            f['properties']['valid'] = False

    out_file = Path(dirs.OUTPUT) / 'predictions' / f'{config_name}_{run_type}_{city}.geojson'
    geofiles.write_json(out_file, census_data)


def correlation_censuslevel(config_name: str, city: str, run_type: str = 'test', scale: str = 'linear'):
    dirs = paths.load_paths()
    pred_file = Path(dirs.OUTPUT) / 'predictions' / f'{config_name}_{run_type}_{city}.geojson'
    if not pred_file.exists():
        run_quantitative_assessment_censuslevel(config_name, city, run_type)
    data = geofiles.load_json(pred_file)
    gts = [f['properties']['POPULATION'] for f in data['features'] if f['properties']['valid']]
    preds = [f['properties']['population_pred'] for f in data['features'] if f['properties']['valid']]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(gts, preds, c='k', s=10, label='Census area')

    slope, intercept, r_value, p_value, std_err = stats.linregress(gts, preds)
    x = np.array([0, 1_000])
    ax.plot(x, slope * x + intercept, c='k')
    # place a text box in upper left in axes coords
    textstr = r'$R^2 = {r_value:.2f}$'.format(r_value=r_value)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=FONTSIZE,
            verticalalignment='top')
    pop_max = 100_000

    ax.plot([0, pop_max], [0, pop_max], c='k', zorder=-1, label='1:1 line')

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
    ax.legend(frameon=False, fontsize=FONTSIZE, loc='upper center')
    out_file = Path(dirs.OUTPUT) / 'plots' / f'{city}_correlation_censuslevel_{config_name}.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.show()


def produce_error_grid(config_name: str, city: str, run_type: str = 'test'):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.CellPopulationDataset(cfg, run_type, no_augmentations=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

    dirs = paths.load_paths()
    pop_file = Path(dirs.DATASET) / 'population_data' / f'pop_{city}.tif'
    pop_arr, transform, crs = geofiles.read_tif(pop_file)

    m, n, _ = pop_arr.shape
    error = np.full((m, n, 3), fill_value=np.NaN, dtype=np.single)
    for item in tqdm(ds):
        x = item['x'].to(device)
        i, j = item['i'], item['j']
        pred_pop = net(x.unsqueeze(0)).flatten().cpu().item()
        pop = item['y'].item()
        error[i, j, 0] = pop - pred_pop
        error[i, j, 1] = pop
        error[i, j, 2] = pred_pop

    out_file = Path(dirs.OUTPUT) / f'error_{config_name}_{run_type}_{city}.tif'
    geofiles.write_tif(out_file, error, transform, crs)


if __name__ == '__main__':
    config = 'resnet18_vhr'
    # qualitative_assessment_celllevel(config)
    # produce_error_grid(config, 'dakar')
    # run_quantitative_assessment_censuslevel(config, 'dakar')
    configs = ['resnet18_bf', 'resnet18_vhr', 'resnet18_vhr4bands', 'resnet18_s2rgb', 'resnet18_s2fc',
               'resnet18_s210m', 'resnet18_s210m_plusbf']
    for config in configs:
        # correlation_celllevel(config, 'dakar')
        correlation_censuslevel(config, 'dakar')
        # produce_error_grid(config, 'dakar')
    # quantitative_assessment(config)