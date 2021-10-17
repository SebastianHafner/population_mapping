import numpy as np
from pathlib import Path
from affine import Affine
from tqdm import tqdm
from utils import geofiles, paths


def preprocess_satellite_data(city: str, patch_size: int):
    dirs = paths.load_paths()
    output_dir = Path(dirs.DATASET) / 'satellite_data' / city
    output_dir.mkdir(exist_ok=True)

    # getting the dimensions of the population file first
    pop_file = Path(dirs.RAW_POPULATION_DATA) / f'population_data_{city}.tif'
    arr_pop, transform_pop, crs_pop = geofiles.read_tif(pop_file)
    height_pop, width_pop, _ = arr_pop.shape
    x_res_pop, _, x_min_pop, _, y_res_pop, y_max_pop, *_ = transform_pop

    pop_file_out = Path(dirs.DATASET) / 'population_data' / f'population_data_{city}.tif'
    pop_file_out.parent.mkdir(exist_ok=True)
    geofiles.write_tif(pop_file_out, arr_pop, transform_pop, crs_pop)

    sat_file = Path(dirs.RAW_SATELLITE_DATA) / f'satellite_data_{city}.tif'
    arr_sat, transform_sat, crs_sat = geofiles.read_tif(sat_file)
    height_sat, width_sat, _ = arr_sat.shape
    x_res_sat, _, x_min_sat, _, y_res_sat, y_max_sat, *_ = transform_sat

    # cropping satellite file to extend of population file
    i_start_sat = int(abs((y_max_sat - y_max_pop) / y_res_sat))
    i_end_sat = int(i_start_sat + abs((height_pop * y_res_pop / y_res_sat)))
    j_start_sat = int((x_min_pop - x_min_sat) // x_res_sat)
    j_end_sat = int(j_start_sat + (width_pop * x_res_pop / x_res_sat))

    arr_sat = arr_sat[i_start_sat:i_end_sat, j_start_sat:j_end_sat, ]
    height_sat, width_sat, _ = arr_sat.shape
    n_rows, n_cols = height_sat // patch_size, width_sat // patch_size

    # earth engine output is row col
    for i in tqdm(range(n_rows)):
        for j in range(n_cols):
            i_start, i_end = i * patch_size, (i + 1) * patch_size
            j_start, j_end = j * patch_size, (j + 1) * patch_size
            tile_arr = arr_sat[i_start:i_end, j_start:j_end, ]

            tile_min_x = x_min_pop + j_start * x_res_sat
            tile_max_y = y_max_pop + i_start * y_res_sat
            tile_transform = Affine(x_res_sat, 0, tile_min_x, 0, y_res_sat, tile_max_y)

            patch_id = f'{i_start:010d}-{j_start:010d}'
            file = output_dir / f'satellite_data_{city}_{patch_id}.tif'
            geofiles.write_tif(file, tile_arr, tile_transform, crs_sat)

    grid_cell_size = int(x_res_pop / x_res_sat)
    metadata = {
        'patch_size': patch_size,  # size of patch in pixels
        'grid_cell_size': grid_cell_size,  # size of a grid cell (assumption square!)
        'samples': [],  # list with all the samples
    }

    for index, population in np.ndenumerate(arr_pop):
        i_pop, j_pop, _ = index
        i_sat, j_sat = int(i_pop * grid_cell_size), int(j_pop * grid_cell_size)

        # use modulo to find patch id
        i_start_patch = int(i_sat % patch_size)
        j_start_patch = int(j_sat % patch_size)

        i_patch = int(i_sat - i_start_patch)
        j_patch = int(j_sat - j_start_patch)

        patch_id = f'{i_patch:010d}-{j_patch:010d}'

        # each pop grid cell is a sample
        metadata['samples'].append({
            'city': city,
            'population': population,
            'patch_id': patch_id,
            'i': i_start_patch,
            'j': j_start_patch,
        })
    metadata_file = Path(dirs.DATASET) / f'metadata_{city}.json'
    geofiles.write_json(metadata_file, metadata)


if __name__ == '__main__':
    # preprocess_satellite_data('dakar', 2048)
    # preprocess_population_data('dakar')
    preprocess_satellite_data('dakar', 1000)


