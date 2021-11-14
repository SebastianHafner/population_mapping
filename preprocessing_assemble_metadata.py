import numpy as np
from pathlib import Path
from utils import geofiles
import argparse


def assemble_metadata(city: str, features: list, raw_data_path: str, dataset_path: str):

    # getting the dimensions of the population file first
    pop_file = Path(raw_data_path) / f'pop_{city}.tif'
    arr_pop, *_ = geofiles.read_tif(pop_file)

    # training-test split according to census units
    train_file = Path(raw_data_path) / f'train_polygons_{city}.tif'
    train_area, _, _ = geofiles.read_tif(train_file)
    test_file = Path(raw_data_path) / f'test_polygons_{city}.tif'
    test_area, _, _ = geofiles.read_tif(test_file)

    # polygons of census areas that are fully covered by population data
    # TODO: this layer could probably also be computed on the fly
    assessment_file = Path(raw_data_path) / f'valid_polygons_{city}.tif'
    assessment_area, _, _ = geofiles.read_tif(assessment_file)

    def get_tiling_data_feature(feature: str) -> dict:
        tiling_data_file = Path(dataset_path) / f'tiling_data_{feature}_{city}.json'
        tiling_data = geofiles.load_json(tiling_data_file)
        return tiling_data
    features_tiling_data = [get_tiling_data_feature(f) for f in features]

    metadata = {
        'features': features,
        'samples': [],  # list with all the samples
    }
    for feature, tiling_data in zip(features, features_tiling_data):
        metadata[f'{feature}_patch_size'] = tiling_data['patch_size']  # size of patch in pixels

    for index, pop in np.ndenumerate(arr_pop):
        i, j, _ = index

        sample = {
            'city': city,
            'population': pop,
            'i': i,
            'j': j,
            'train_poly': int(train_area[i, j, 0]),
            'test_poly': int(test_area[i, j, 0]),
            'valid_for_assessment': int(assessment_area[i, j, 0]),
        }

        for feature, tiling_data in zip(features, features_tiling_data):
            patch_id = f'{i:03d}-{j:03d}'
            patch_file = Path(dataset_path) / 'features' / city / feature / f'{feature}_{city}_{patch_id}.tif'
            assert(patch_file.exists())

        metadata['samples'].append(sample)

    metadata_file = Path(dataset_path) / f'metadata_{city}.json'
    geofiles.write_json(metadata_file, metadata)


def metadata_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--city", dest='city', required=True)
    parser.add_argument('-f', "--feature", dest='features', action='append', required=True)
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


