import numpy as np
from pathlib import Path
from utils import geofiles, parsers
import geopandas as gpd


# -f vhr s2 bf -r C:/Users/shafner/population_mapping/raw_data -d C:/Users/shafner/datasets/pop_dataset
def assemble_metadata(features: list, raw_data_path: str, dataset_path: str):
    city = 'dakar'
    # getting the dimensions of the population file first
    pop_file = Path(raw_data_path) / city / f'pop_{city}.tif'
    pop, *_ = geofiles.read_tif(pop_file)
    pop_is_nan = np.isnan(pop)

    metadata = {
        'features': features,
        'census': {},
        'split': {},
        'samples': [],  # list with all the samples
    }

    for split in ['train', 'test']:
        census_file = Path(raw_data_path) / city / f'{split}_{city}.shp'
        gdf = gpd.read_file(census_file)
        for index, row in gdf.iterrows():
            unit_nr = row['cat']
            unit_pop = row['Dakar_St_5']
            metadata['census'][unit_nr] = unit_pop
            metadata['split'][unit_nr] = split
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
                    assert (patch_file.exists())
                metadata['samples'].append(sample)

    metadata_file = Path(dataset_path) / f'metadata_{city}.json'
    geofiles.write_json(metadata_file, metadata)


if __name__ == '__main__':
    args = parsers.metadata_argument_parser().parse_known_args()[0]
    assemble_metadata(args.features, args.rawdata_dir, args.dataset_dir)

