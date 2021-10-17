import yaml
from pathlib import Path
from utils import experiment_manager

# set the paths
HOME = '/home/shafner/population_mapping'
DATASET = '/storage/shafner/population_mapping/pop_dataset'
OUTPUT = '/storage/shafner/population_mapping_output'
RAW_SATELLITE_DATA = '/storage/shafner/population_mapping/raw_satellite_data'
RAW_POPULATION_DATA = '/storage/shafner/population_mapping/raw_population_data'


# TODO: define return type as cfg node from experiment manager
def load_paths():
    C = experiment_manager.CfgNode()
    C.HOME = HOME
    C.DATASET = DATASET
    C.OUTPUT = OUTPUT
    C.RAW_SATELLITE_DATA = RAW_SATELLITE_DATA
    C.RAW_POPULATION_DATA = RAW_POPULATION_DATA
    return C.clone()


def setup_directories():
    dirs = load_paths()

    # inference dir
    inference_dir = Path(dirs.OUTPUT) / 'inference'
    inference_dir.mkdir(exist_ok=True)

    # evaluation dirs
    evaluation_dir = Path(dirs.OUTPUT) / 'evaluation'
    evaluation_dir.mkdir(exist_ok=True)
    quantiative_evaluation_dir = evaluation_dir / 'quantitative'
    quantiative_evaluation_dir.mkdir(exist_ok=True)
    qualitative_evaluation_dir = evaluation_dir / 'qualitative'
    qualitative_evaluation_dir.mkdir(exist_ok=True)

    # testing
    testing_dir = Path(dirs.OUTPUT) / 'testing'
    testing_dir.mkdir(exist_ok=True)

    # saving networks
    networks_dir = Path(dirs.OUTPUT) / 'networks'
    networks_dir.mkdir(exist_ok=True)

    # plots
    plots_dir = Path(dirs.OUTPUT) / 'plots'
    plots_dir.mkdir(exist_ok=True)



if __name__ == '__main__':
    setup_directories()
