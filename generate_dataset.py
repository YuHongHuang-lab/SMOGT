import yaml

from dataset.bio_dataset import BioDataset

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


config = load_config('config.yaml')
dataset = BioDataset(config=config["dataset"])
