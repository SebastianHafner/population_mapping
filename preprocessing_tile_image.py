from pathlib import Path
from affine import Affine
from tqdm import tqdm
from utils import geofiles
import argparse

# x min, x max, y min, y max
# extent dakar [227600, 256500, 1620600, 1639800]


# split a feature (e.g. vhr satellite data) into patches
def tile_image(city: str, feature: str, extent: list, patch_size: int, raw_data_path: str, dataset_path: str):
    output_dir = Path(dataset_path) / 'features' / city / feature
    output_dir.mkdir(exist_ok=True)

    extent = [float(n) for n in extent]
    x_min_ref, x_max_ref, y_min_ref, y_max_ref = extent

    file = Path(raw_data_path) / city / f'{feature}_{city}.tif'
    arr, transform, crs = geofiles.read_tif(file)
    height, width, _ = arr.shape
    x_res, _, x_min, _, y_res, y_max, *_ = transform

    # cropping feature file to extent of population file
    j_start = int((x_min_ref - x_min) // x_res)
    n = int((x_max_ref - x_min_ref) / x_res)
    j_end = int(j_start + n)

    i_start = int(abs((y_max - y_max_ref) / y_res))
    m = int(abs((y_max_ref - y_min_ref) / y_res))
    i_end = int(i_start + m)

    arr = arr[i_start:i_end, j_start:j_end, ]
    height, width, _ = arr.shape

    # patch size in m (m and n should be the same because square)
    patch_size = float(patch_size)
    m_patch = int(patch_size / abs(y_res))
    n_patch = int(patch_size / x_res)
    assert(m_patch == n_patch)

    n_rows, n_cols = int(height / m_patch), int(width / n_patch)
    for i in tqdm(range(n_rows)):
        for j in range(n_cols):
            i_start, i_end = i * m_patch, (i + 1) * m_patch
            j_start, j_end = j * n_patch, (j + 1) * n_patch
            patch = arr[i_start:i_end, j_start:j_end, ]

            x_min_patch = x_min_ref + j_start * x_res
            y_max_patch = y_max_ref + i_start * y_res
            transform_patch = Affine(x_res, 0, x_min_patch, 0, y_res, y_max_patch)

            patch_id = f'{i:03d}-{j:03d}'
            file = output_dir / f'{feature}_{city}_{patch_id}.tif'
            geofiles.write_tif(file, patch, transform_patch, crs)


def tiling_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--city", dest='city', required=True)
    parser.add_argument('-f', "--feature", dest='feature', required=True)
    parser.add_argument('-e' "--extent", nargs="+", dest='extent', required=True)
    parser.add_argument('-p' "--patch-size", dest='patch_size', required=True)
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
    tile_image(args.city, args.feature, args.extent, args.patch_size, args.rawdata_dir, args.dataset_dir)



