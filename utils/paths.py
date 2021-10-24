import yaml
from pathlib import Path
from utils import experiment_manager

# set the paths
HOME = 'C:/Users/shafner/repos/population_mapping'  # '/home/shafner/population_mapping'
DATASET = 'C:/Users/shafner/datasets/pop_dataset'  # '/storage/shafner/population_mapping/pop_dataset'
OUTPUT = 'C:/Users/shafner/population_mapping/output'  # '/storage/shafner/population_mapping_output'
RAW_DATA = 'C:/Users/shafner/population_mapping/raw_data'  # '/storage/shafner/population_mapping/raw_data'


# TODO: define return type as cfg node from experiment manager
def load_paths():
    C = experiment_manager.CfgNode()
    C.HOME = HOME
    C.DATASET = DATASET
    C.OUTPUT = OUTPUT
    C.RAW_DATA = RAW_DATA
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
