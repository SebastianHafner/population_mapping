import torch
from torch.utils import data as torch_data
import numpy as np
import wandb
from tqdm import tqdm
from utils import datasets, metrics, experiment_manager, networks


def model_evaluation(net: networks.CustomNet, cfg: experiment_manager.CfgNode, run_type: str, epoch: float, step: int,
                     max_samples: int = 100):

    measurer = RegressionEvaluation()

    dataset = datasets.PopulationMappingDataset(cfg, run_type, no_augmentations=True)

    def evaluation_callback(x, y, z):
        # x img y label z logits
        measurer.add_sample(z, y)

    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    inference_loop(net, cfg, dataset, evaluation_callback, max_samples=max_samples, num_workers=num_workers)

    # assessment
    rmse = measurer.root_mean_square_error()
    print(f'RMSE {run_type} {rmse:.3f}')
    if not cfg.DEBUG:
        wandb.log({
            f'{run_type} rmse': rmse,
            'step': step,
            'epoch': epoch,
        })


def inference_loop(net: networks.CustomNet, cfg: experiment_manager.CfgNode, dataset: str, callback,
                   max_samples: int = None, num_workers: int = 0):
    dataloader_kwargs = {
        'batch_size': 1,
        'num_workers': num_workers,
        'shuffle': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    max_samples = len(dataset) if max_samples is None else max_samples

    counter = 0
    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(dataloader):
            img = batch['x'].to(device)
            label = batch['y'].to(device)

            logits = net(img)

            callback(img, label, logits)

            counter += 1
            if counter == max_samples or cfg.DEBUG:
                break


class RegressionEvaluation(object):
    def __init__(self):
        self.predictions = []
        self.labels = []

    def add_sample(self, logits: torch.tensor, label: torch.tensor):

        pred = torch.sigmoid(logits)
        pred = pred.float().detach().cpu().numpy()

        label = label.float().detach().cpu().numpy()

        self.predictions.extend(pred.flatten())
        self.labels.extend(label.flatten())

    def reset(self):
        self.predictions = []
        self.labels = []

    def root_mean_square_error(self) -> float:
        return np.sqrt(np.sum(np.square(np.array(self.predictions) - np.array(self.labels))) / len(self.labels))

