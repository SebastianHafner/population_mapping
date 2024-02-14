from pathlib import Path
from affine import Affine
from tqdm import tqdm
from utils import geofiles, parsers

# pleidas https://docs.sentinel-hub.com/api/latest/data/airbus/pleiades/
# worldview-3 http://worldview3.digitalglobe.com/
# x min, x max, y min, y max
# extent dakar [227600, 256500, 1620600, 1639800]
# extent nairobi [240100, 289150, 9840110, 9871640]
# extent daressalaam [508530, 542730, 9226610, 9268710] new [509630, 542730, 9231810, 9268710]
# extent ouagadougou [645070, 671170, 1355620, 1382420]
# extent ndjamena [488170 519470 1325680 1350580]
# extent ouaddai [462370, 518860, 1516200, 1562000]
# -c nairobi -f s2 -e 240100 289200 9840100 9871700 -p 100 -r C:/Users/shafner/population_mapping/raw_data -d C:/Users/shafner/datasets/pop_dataset
# -c dakar -f vhr -e 227600 256500 1620600 1639800 -p 100 -r C:/Users/shafner/population_mapping/raw_data -d C:/Users/shafner/datasets/pop_dataset
# -c daressalaam -f s2 -e 509630 542730 9231810 9268710 -p 100 -r C:/Users/shafner/population_mapping/raw_data -d C:/Users/shafner/datasets/pop_dataset
# -c ouagadougou -f s2 -e 645070 671170 1355620 1382420 -p 100 -r C:/Users/shafner/population_mapping/raw_data -d C:/Users/shafner/datasets/pop_dataset
# -c ndjamena -f s2 -e 488170 519470 1325680 1350580 -p 100 -r C:/Users/shafner/population_mapping/raw_data -d C:/Users/shafner/datasets/pop_dataset
# -c ouaddai -f s2 -e 462370 518860 1516200 1562000 -p 100 -r C:/Users/shafner/population_mapping/raw_data -d C:/Users/shafner/datasets/pop_dataset
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


if __name__ == '__main__':
    args = parsers.tiling_argument_parser().parse_known_args()[0]
    tile_image(args.city, args.feature, args.extent, args.patch_size, args.rawdata_dir, args.dataset_dir)



