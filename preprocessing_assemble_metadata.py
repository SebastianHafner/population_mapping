import numpy as np
from pathlib import Path
from utils import geofiles
import argparse


def assemble_metadata(city: str, features: list, raw_data_path: str, dataset_path: str):

    # getting the dimensions of the population file first
    pop_file = Path(raw_data_path) / city / f'pop_{city}.tif'
    pop, *_ = geofiles.read_tif(pop_file)
    pop_is_nan = np.isnan(pop)

    metadata = {
        'features': features,
        'samples': [],  # list with all the samples
    }

    for split in ['train', 'test']:
        units_file = Path(raw_data_path) / city / f'{split}_{city}.tif'
        units, _, _ = geofiles.read_tif(units_file)

        for index, unit in np.ndenumerate(units):
            if unit != 0:
                i, j, _ = index
                sample = {
                    'city': city,
                    'population': 0 if pop_is_nan[index] else float(pop[index]),
                    'i': i,
                    'j': j,
                    'split': split,
                    'unit': int(unit),
                }

                patch_id = f'{i:03d}-{j:03d}'
                for feature in features:
                    patch_file = Path(dataset_path) / 'features' / city / feature / f'{feature}_{city}_{patch_id}.tif'
                    assert(patch_file.exists())
                metadata['samples'].append(sample)

    metadata_file = Path(dataset_path) / f'metadata_{city}.json'
    geofiles.write_json(metadata_file, metadata)


def metadata_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--city", dest='city', required=True)
    parser.add_argument('-f' "--feature", nargs="+", dest='features', required=True)
    parser.add_argument('-r', "--rawdata-dir", dest='rawdata_dir', required=True, help="path to raw data directory")
    parser.add_argument('-d', "--dataset-dir", dest='dataset_dir', default="", required=True,
                        help="path to dataset directory")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = metadata_argument_parser().parse_known_args()[0]
    assemble_metadata(args.city, args.features, args.rawdata_dir, args.dataset_dir)


