from pathlib import Path
from utils import paths, geofiles
import numpy as np
from tqdm import tqdm


def population_stats(city: str):
    dirs = paths.load_paths()
    pop_file = Path(dirs.DATASET) / 'population_data' / f'pop_{city}.tif'
    pop_data, *_ = geofiles.read_tif(pop_file)
    print(np.nanmax(pop_data))
    # TODO: add boxplot with distribution
    pass


def vhr_stats(city: str):
    dirs = paths.load_paths()
    metadata_file = Path(dirs.DATASET) / f'metadata_{city}.json'
    metadata = geofiles.load_json(metadata_file)
    samples = metadata['samples']
    data = []
    for i, sample in enumerate(tqdm(samples)):
        patch_id = sample['patch_id']
        patch_file = Path(dirs.DATASET) / 'satellite_data' / city / f'vhr_{city}_{patch_id}.tif'
        arr, _, _ = geofiles.read_tif(patch_file)
        data.append(arr)
    data = np.stack(data).flatten()
    print(f'min: {np.min(data)}')
    print(f'max: {np.max(data)}')
    print(f'mean: {np.mean(data)}')
    print(f'std: {np.std(data)}')


if __name__ == '__main__':
    # population_stats('dakar')
    vhr_stats('dakar')
