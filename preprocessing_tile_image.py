from pathlib import Path
from affine import Affine
from tqdm import tqdm
from utils import geofiles, paths
import argparse


# split a feature (e.g. vhr satellite data) into patches according to extend of pop file and patch size
# also outputs a metadata file for the tiling
def tile_image(city: str, feature: str, raw_data_path: str, dataset_path: str):
    output_dir = Path(dataset_path) / 'features' / city / feature
    output_dir.mkdir(exist_ok=True)

    # getting the dimensions of the population file first
    pop_file = Path(raw_data_path) / f'pop_{city}.tif'
    arr_pop, transform_pop, crs_pop = geofiles.read_tif(pop_file)
    height_pop, width_pop, _ = arr_pop.shape
    x_res_pop, _, x_min_pop, _, y_res_pop, y_max_pop, *_ = transform_pop

    file = Path(raw_data_path) / f'{feature}_{city}.tif'
    arr, transform, crs = geofiles.read_tif(file)
    height, width, _ = arr.shape
    x_res, _, x_min, _, y_res, y_max, *_ = transform

    # cropping feature file to extend of population file
    i_start = int(abs((y_max - y_max_pop) / y_res))
    i_end = int(i_start + abs((height_pop * y_res_pop / y_res)))
    j_start = int((x_min_pop - x_min) // x_res)
    j_end = int(j_start + (width_pop * x_res_pop / x_res))

    arr = arr[i_start:i_end, j_start:j_end, ]
    height, width, _ = arr.shape
    patch_size = int(x_res_pop / x_res)

    n_rows, n_cols = height // patch_size, width // patch_size

    tiling_data = {
        'feature': feature,
        'patch_size': patch_size,  # size of patch in pixels
    }

    # earth engine output is row col
    assert (n_rows == height_pop and n_cols == width_pop)
    for i in tqdm(range(n_rows)):
        for j in range(n_cols):
            i_start, i_end = i * patch_size, (i + 1) * patch_size
            j_start, j_end = j * patch_size, (j + 1) * patch_size
            patch = arr[i_start:i_end, j_start:j_end, ]

            x_min_patch = x_min_pop + j_start * x_res
            y_max_patch = y_max_pop + i_start * y_res
            transform_patch = Affine(x_res, 0, x_min_patch, 0, y_res, y_max_patch)

            patch_id = f'{i:03d}-{j:03d}'
            file = output_dir / f'{feature}_{city}_{patch_id}.tif'
            geofiles.write_tif(file, patch, transform_patch, crs)

    tiling_data_file = Path(dataset_path) / f'tiling_data_{feature}_{city}.json'
    geofiles.write_json(tiling_data_file, tiling_data)


def tiling_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--city", dest='city', required=True)
    parser.add_argument('-f', "--feature", dest='feature', required=True)
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
    args = tiling_argument_parser().parse_known_args()[0]
    tile_image(args.city, args.feature, args.rawdata_dir, args.dataset_dir)



