from pathlib import Path
import geopandas as gpd
import argparse


def create_census_geojson(city: str, run_type: str, raw_data_path: str, dataset_path: str):
    file = Path(raw_data_path) / f'{run_type}ing_polygons.shp'
    gdf = gpd.read_file(file)
    gdf['poly_id'] = gdf['cluster'] + 1
    gdf = gdf[['POPULATION', 'poly_id', 'geometry']]
    out_file = Path(dataset_path) / 'census_data' / f'{run_type}_polygons_{city}.geojson'
    gdf.to_file(out_file, driver='GeoJSON')


def census_argument_parser():
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method
    parser = argparse.ArgumentParser(description="Experiment Args")
    parser.add_argument('-c', "--city", dest='city', required=True)
    parser.add_argument('-t', "--run_type", dest='run_type', required=True)
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
    args = census_argument_parser().parse_known_args()[0]
    create_census_geojson(args.city, args.run_type, args.rawdata_dir, args.dataset_dir)