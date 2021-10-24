from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from utils import paths, geofiles, datasets, experiment_manager


def check_metadata_file(city: str):
    dirs = paths.load_paths()
    metadata_file = Path(dirs.DATASET) / f'metadata_{city}.json'
    metadata = geofiles.load_json(metadata_file)
    for s in metadata['samples']:
        patch_id = s['patch_id']
        file = Path(dirs.DATASET) / 'satellite_data' / city / f'vhr_{city}_{patch_id}.tif'
        if not file.exists():
            print(file.name)


def check_dataset(config_name: str, n_samples: int = 5):
    cfg = experiment_manager.load_cfg(config_name)
    ds = datasets.PopulationMappingDataset(cfg, 'training', no_augmentations=True)
    scale_factor = 0.5
    indices = np.random.randint(0, len(ds), n_samples)
    for index in indices:
        item = ds.__getitem__(index)
        img, y = item['x'], item['y']

        fig, ax = plt.subplots(1, 1)
        bands = [0, 1, 2]
        img = img[bands, ].numpy().transpose((1, 2, 0))
        img = np.clip(img / scale_factor, 0, 1)
        population = y * cfg.DATALOADER.POP_GRIDCELL_MAX

        ax.imshow(img)
        ax.set_title(f'y: {y:.2f}; pop: {population:.0f}')
        ax.set_axis_off()
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    # check_metadata_file('dakar')
    check_dataset('debug')