from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import parsers, geofiles, visualization


def population_stats(city: str):
    dirs = paths.load_paths()
    pop_file = Path(dirs.DATASET) / 'population_data' / f'pop_{city}.tif'
    pop_data, *_ = geofiles.read_tif(pop_file)
    print(np.nanmax(pop_data))
    # TODO: add boxplot with distribution
    pass


def vhr_stats(city: str, dataset_path):
    metadata_file = Path(dataset_path) / f'metadata_{city}.json'
    metadata = geofiles.load_json(metadata_file)
    samples = metadata['samples']
    data = []
    for i, sample in enumerate(tqdm(samples)):
        patch_id = sample['patch_id']
        patch_file = Path(dataset_path) / 'satellite_data' / city / f'vhr_{city}_{patch_id}.tif'
        arr, _, _ = geofiles.read_tif(patch_file)
        data.append(arr)
    data = np.stack(data).flatten()
    print(f'min: {np.min(data)}')
    print(f'max: {np.max(data)}')
    print(f'mean: {np.mean(data)}')
    print(f'std: {np.std(data)}')


def plot_vhr_distribution(city: str, dataset_path):
    pass


def visualize_feature(feature: str, city: str, dataset_path: str):
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
    args = parsers.inspector_argument_parser().parse_known_args()[0]
    # population_stats('dakar')
    visualize_feature(args.feature, args.city, args.dataset_dir)
    # vhr_stats('dakar')
