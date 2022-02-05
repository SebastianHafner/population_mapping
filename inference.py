from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import parsers, geofiles, experiment_manager, datasets, networks
import torch


def produce_population_grid(cfg: experiment_manager.CfgNode, city: str):
    ds = datasets.CellInferencePopulationDataset(cfg, city)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()
    arr = ds.get_arr()
    transform, crs = ds.get_geo()
    for item in tqdm(ds):
        x = item['x'].to(device)
        i, j = item['i'], item['j']
        pred_pop = net(x.unsqueeze(0)).flatten().cpu().item()
        arr[i, j, 0] = pred_pop
    out_file = Path(cfg.PATHS.OUTPUT) / 'inference' / f'pop_{city}_{cfg.NAME}.tif'
    geofiles.write_tif(out_file, arr, transform, crs)


def produce_error_grid(config_name: str, city: str, dataset_path: str, output_path: str):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.CellPopulationDataset(cfg, run_type, no_augmentations=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, *_ = networks.load_checkpoint(cfg.INFERENCE_CHECKPOINT, cfg, device)
    net.eval()

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
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    produce_population_grid(cfg, args.site)
