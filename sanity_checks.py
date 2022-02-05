from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from utils import geofiles, datasets, experiment_manager, parsers, visualization


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


def check_features(features: list, city: str, dataset_path: str):
    metadata_file = Path(dataset_path) / f'metadata_{city}.json'
    metadata = geofiles.load_json(metadata_file)
    samples = metadata['samples']
    order = list(np.random.rand(len(samples)))
    samples_random = [s for _, s in sorted(zip(order, samples), key=lambda pair: pair[0])]

    plot_size = 3
    m, n = 10, 10
    fig, axs = plt.subplots(m, n, figsize=(n * plot_size, m * plot_size))
    plt.tight_layout()
    for i, ax in enumerate(list(axs.flatten())):
        sample = samples_random[i]
        i, j = sample['i'], sample['j']
        if feature == 'vhr':
            visualization.plot_vhr(ax, dataset_path, city, i, j)
        elif feature == 's2':
            visualization.plot_s2(ax, dataset_path, city, i, j)
        elif feature == 'bf':
            visualization.plot_bf(ax, dataset_path, city, i, j)
        else:
            raise Exception(f'{feature} unknown feature')
    plt.show()


if __name__ == '__main__':
    # check_metadata_file('dakar')
    check_dataset('debug')