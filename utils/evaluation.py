import torch
from torch.utils import data as torch_data
import numpy as np
import wandb
from tqdm import tqdm
from utils import datasets, experiment_manager, networks, geofiles
from scipy import stats
from pathlib import Path


class RegressionEvaluation(object):
    def __init__(self):
        self.predictions = []
        self.labels = []

    def add_sample(self, pred: torch.tensor, label: torch.tensor):
        pred = pred.float().detach().cpu().numpy()
        label = label.float().detach().cpu().numpy()
        self.predictions.extend(pred.flatten())
        self.labels.extend(label.flatten())

    def reset(self):
        self.predictions = []
        self.labels = []

    def root_mean_square_error(self) -> float:
        return np.sqrt(np.sum(np.square(np.array(self.predictions) - np.array(self.labels))) / len(self.labels))


def model_evaluation_cell(net: networks.PopulationNet, cfg: experiment_manager.CfgNode, run_type: str, epoch: float,
                          step: int, max_samples: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    measurer = RegressionEvaluation()
    dataset = datasets.CellPopulationDataset(cfg, run_type, no_augmentations=True)
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    max_samples = len(dataset) if max_samples is None else max_samples
    counter = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            img = batch['x'].to(device)
            label = batch['y'].to(device)
            pred = net(img)
            measurer.add_sample(pred, label)
            counter += 1
            if counter == max_samples or cfg.DEBUG:
                break

    # assessment
    rmse = measurer.root_mean_square_error()
    print(f'RMSE {run_type} {rmse:.3f}')
    wandb.log({
        f'{run_type} rmse': rmse,
        'step': step,
        'epoch': epoch,
    })


def model_evaluation_census(net: networks.PopulationNet, cfg: experiment_manager.CfgNode, city: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    measurer = RegressionEvaluation()

    metadata_file = Path(cfg.PATHS.DATASET) / f'metadata_{city}.json'
    metadata = geofiles.load_json(metadata_file)
    census = metadata['census']

    gt_units = []
    pred_units = []
    for unit_nr, unit_gt in tqdm(census.items()):
        unit_nr, unit_gt = int(unit_nr), int(unit_gt)
        ds = datasets.CensusPopulationDataset(cfg, city, unit_nr)
        if ds.split == 'train':
            continue
        unit_pred = 0
        for i, index in enumerate(range(len(ds))):
            item = ds.__getitem__(index)
            x = item['x'].to(device)
            y = item['y'].to(device)

            pop_pred = net(x.to(device).unsqueeze(0))
            measurer.add_sample(pop_pred, y)
            unit_pred += pop_pred.cpu().item()
            unit_gt += item['y'].cpu().item()
        gt_units.append(unit_gt)
        pred_units.append(unit_pred)
    rmse = measurer.root_mean_square_error()
    slope, intercept, r_value, p_value, std_err = stats.linregress(gt_units, pred_units)
    wandb.log({
        f'{city} rmse': rmse,
        f'{city} r2': r_value,
    })


def model_evaluation_cell_dualstream(dual_net: networks.DualStreamPopulationNet, dual_cfg: experiment_manager.CfgNode,
                                     run_type: str, epoch: float, step: int, max_samples: int = None):
    measurer_stream1 = RegressionEvaluation()
    measurer_stream2 = RegressionEvaluation()
    measurer_fusion = RegressionEvaluation()

    dataset = datasets.CellDualInputPopulationDataset(dual_cfg, run_type, no_augmentations=True)

    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': 0 if dual_cfg.DEBUG else dual_cfg.CFG1.DATALOADER.NUM_WORKER,
        'shuffle': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dual_net.to(device)
    max_samples = len(dataset) if max_samples is None else max_samples

    counter = 0
    with torch.no_grad():
        dual_net.eval()
        for step, batch in enumerate(tqdm(dataloader)):
            x1 = batch['x1'].to(device)
            x2 = batch['x2'].to(device)
            label = batch['y'].to(device)

            pred_fusion, pred_stream1, pred_stream2 = dual_net(x1, x2)
            measurer_stream1.add_sample(pred_stream1, label)
            measurer_stream2.add_sample(pred_stream2, label)
            measurer_fusion.add_sample(pred_fusion, label)

            counter += 1
            if counter == max_samples or dual_cfg.DEBUG:
                break

    # assessment
    rmse_stream1 = measurer_stream1.root_mean_square_error()
    rmse_stream2 = measurer_stream2.root_mean_square_error()
    rmse_fusion = measurer_fusion.root_mean_square_error()
    print(f'RMSE {run_type} {rmse_fusion:.3f}')
    wandb.log({
        f'{run_type} rmse': rmse_fusion,
        f'{run_type} rmse_stream1': rmse_stream1,
        f'{run_type} rmse_stream2': rmse_stream2,
        'step': step,
        'epoch': epoch,
    })
