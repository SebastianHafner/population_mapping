from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import parsers, geofiles, experiment_manager, datasets, networks
import torch





if __name__ == '__main__':
    args = parsers.inference_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    for city in args.sites:
        produce_population_grid(cfg, city)
