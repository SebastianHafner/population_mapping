import numpy as np
from pathlib import Path
from affine import Affine
import geopandas as gpd
from tqdm import tqdm
from utils import geofiles, paths


def run_preprocessing(city: str, patch_size: int):
    dirs = paths.load_paths()
    output_dir = Path(dirs.DATASET) / 'satellite_data' / city
    output_dir.mkdir(exist_ok=True)

    # getting the dimensions of the population file first
    pop_file = Path(dirs.RAW_DATA) / f'pop_{city}.tif'
    arr_pop, transform_pop, crs_pop = geofiles.read_tif(pop_file)
    height_pop, width_pop, _ = arr_pop.shape
    x_res_pop, _, x_min_pop, _, y_res_pop, y_max_pop, *_ = transform_pop

    sat_file = Path(dirs.RAW_DATA) / f'vhr_{city}.tif'
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

    grid_cell_size = int(x_res_pop / x_res_sat)
    grids_per_patch = patch_size // grid_cell_size
    metadata = {
        'patch_size': patch_size,  # size of patch in pixels
        'grid_cell_size': grid_cell_size,  # size of a grid cell (assumption square!)
        'samples': [],  # list with all the samples
    }

    train_file = Path(dirs.RAW_DATA) / f'train_polygons_{city}.tif'
    train_area, _, _ = geofiles.read_tif(train_file)
    test_file = Path(dirs.RAW_DATA) / f'test_polygons_{city}.tif'
    test_area, _, _ = geofiles.read_tif(test_file)
    assessment_file = Path(dirs.RAW_DATA) / f'valid_polygons_{city}.tif'
    assessment_area, _, _ = geofiles.read_tif(assessment_file)

    # earth engine output is row col
    # loop over patches
    for i in tqdm(range(n_rows)):
        for j in range(n_cols):
            i_start, i_end = i * patch_size, (i + 1) * patch_size
            j_start, j_end = j * patch_size, (j + 1) * patch_size
            arr_patch = arr_sat[i_start:i_end, j_start:j_end, ]

            x_min_patch = x_min_pop + j_start * x_res_sat
            y_max_patch = y_max_pop + i_start * y_res_sat
            transform_patch = Affine(x_res_sat, 0, x_min_patch, 0, y_res_sat, y_max_patch)

            patch_id = f'{i_start:010d}-{j_start:010d}'
            file = output_dir / f'vhr_{city}_{patch_id}.tif'
            geofiles.write_tif(file, arr_patch, transform_patch, crs_sat)

            # loop over grid of a patch
            for i_grid in range(grids_per_patch):
                for j_grid in range(grids_per_patch):
                    i_pop = i * grids_per_patch + i_grid
                    j_pop = j * grids_per_patch + j_grid

                    # each pop grid cell is a sample
                    metadata['samples'].append({
                        'city': city,
                        'population': float(arr_pop[i_pop, j_pop, 0]),
                        'train_poly': int(train_area[i_pop, j_pop, 0]),
                        'test_poly': int(test_area[i_pop, j_pop, 0]),
                        'valid_for_assessment': int(assessment_area[i_pop, j_pop, 0]),
                        'patch_id': patch_id,
                        'i': i_grid * grid_cell_size,
                        'j': j_grid * grid_cell_size,
                    })

    metadata_file = Path(dirs.DATASET) / f'metadata_{city}.json'
    geofiles.write_json(metadata_file, metadata)


def create_census_geojson(city: str, run_type: str):
    dirs = paths.load_paths()
    file = Path(dirs.RAW_DATA) / f'{run_type}ing_polygons.shp'
    gdf = gpd.read_file(file)
    gdf['poly_id'] = gdf['cluster'] + 1
    gdf = gdf[['POPULATION', 'poly_id', 'geometry']]
    out_file = Path(dirs.DATASET) / 'census_data' / f'{run_type}_polygons_{city}.geojson'
    gdf.to_file(out_file, driver='GeoJSON')



if __name__ == '__main__':
    # run_preprocessing('dakar', 1000)
    for rt in ['train', 'test']:
        create_census_geojson('dakar', rt)


