from torch_geometric.data import Data

from dataset.bio_dataset import BioDataset
from utils import *
import time


config_path = "config.yaml"
config = load_config(config_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO if config["log"]["level"] == "INFO" else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset and split edges
dataset = BioDataset(config=config["dataset"])
data = dataset.data

# data = dataset.to_homegeneous()
data = split_edges(
    data,
    rate=config['dataset'].get('rate', 0.2),
    seed=config['dataset'].get('seed', 42),
    mask_prior=config['dataset'].get('mask_prior', False),
    prior_mask_rate=0,
    use_last_col=False
)

node_label_dict = {}
for node_type in data.node_types:
    node_label_dict[node_type] = {
        i: label for i, label in enumerate(data[node_type].label)
    }

# Create combined data with both train and test edges
combined_data_base = Data(
    x_dict=data.x_dict,
    prior_edge_index_dict={edge_type: data[edge_type].edge_index
                           for edge_type in data.edge_types},
    prior_edge_attr_dict={edge_type: data[edge_type].edge_attr
                          for edge_type in data.edge_types},
    global_edge_index_dict={edge_type: data[edge_type].global_edge_index
                            for edge_type in data.edge_types
                            if hasattr(data[edge_type], 'global_edge_index')},
    global_edge_attr_dict={edge_type: data[edge_type].global_edge_attr
                           for edge_type in data.edge_types
                           if hasattr(data[edge_type], 'global_edge_attr')},
    train_edge_index_dict={edge_type: data[edge_type].train_edge_index
                           for edge_type in data.edge_types
                           if hasattr(data[edge_type], 'train_edge_index')},
    test_edge_index_dict={edge_type: data[edge_type].test_edge_index
                          for edge_type in data.edge_types
                          if hasattr(data[edge_type], 'test_edge_index')},
    add_edge_pool_dict={
        edge_type: torch.cat([
            data[edge_type].train_edge_index,
            data[edge_type].test_edge_index
        ], dim=1)
        for edge_type in data.edge_types
        if hasattr(data[edge_type], 'train_edge_index') and hasattr(data[edge_type], 'test_edge_index')
    }
)

# Pre-generate data with negative samples for both train and test
num_epochs = config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
negative_rate = config["model"]["Negative_rate"]
# negative_rate = 3

# data_list = joblib.load(filename=f"{config['dataset']['processed_dir']}data_list_hetero_1000.pkl")
data_list = []
start_time = time.time()
data = create_data_with_neg_samples_parallel(
    combined_data_base,
    device,
    negative_rate,
    40,
    # num_workers=config['dataset'].get('ncores', 10)
    num_workers=20,
    node_label_dict=node_label_dict
)
end_time = time.time()
print(f"代码执行耗时: {end_time - start_time:.2f} 秒")
data_list.extend(data)
# data_list = data_list[0:120]
joblib.dump(data_list, filename=f"{config['dataset']['processed_dir']}data_list_hetero_2.pkl")

#bantc 1
joblib.dump(data_list[:100], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_1.pkl")
joblib.dump(data_list[101:200], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_2.pkl")
joblib.dump(data_list[201:300], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_3.pkl")
joblib.dump(data_list[301:400], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_4.pkl")

#bantch 2
joblib.dump(data_list[:100], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_5.pkl")
joblib.dump(data_list[101:200], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_6.pkl")
joblib.dump(data_list[201:300], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_7.pkl")
joblib.dump(data_list[301:400], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_8.pkl")

#bantch 3
joblib.dump(data_list[:100], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_9.pkl")
joblib.dump(data_list[101:200], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_10.pkl")
joblib.dump(data_list[201:300], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_11.pkl")
joblib.dump(data_list[301:400], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_12.pkl")

#bantch 4
joblib.dump(data_list[:100], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_13.pkl")
joblib.dump(data_list[101:200], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_14.pkl")
joblib.dump(data_list[201:300], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_15.pkl")
joblib.dump(data_list[301:400], filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_16.pkl")

joblib.dump(data_list, filename="/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/data_list_hetero_5.pkl")



# test
# if __name__ == '__main__':
#     config_path = "config.yaml"
#     config = load_config(config_path)
#
#     # Setup logging
#     logging.basicConfig(
#         level=logging.INFO if config["log"]["level"] == "INFO" else logging.DEBUG,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
#     logging.info(config)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Create dataset and split edges
#     dataset = BioDataset(config=config["dataset"])
#     data = dataset.data
#
#     # data = dataset.to_homegeneous()
#     data = split_edges(
#         data,
#         rate=config['dataset'].get('rate', 0.2),
#         seed=config['dataset'].get('seed', 42),
#         mask_prior=config['dataset'].get('mask_prior', False),
#         prior_mask_rate=0,
#         use_last_col=False
#     )
#
#     node_label_dict = {}
#     for node_type in data.node_types:
#         node_label_dict[node_type] = {
#             i: label for i, label in enumerate(data[node_type].label)
#         }
#
#     # Create combined data with both train and test edges
#     combined_data_base = Data(
#         x_dict=data.x_dict,
#         prior_edge_index_dict={edge_type: data[edge_type].edge_index
#                                for edge_type in data.edge_types},
#         prior_edge_attr_dict={edge_type: data[edge_type].edge_attr
#                               for edge_type in data.edge_types},
#         global_edge_index_dict={edge_type: data[edge_type].global_edge_index
#                                 for edge_type in data.edge_types
#                                 if hasattr(data[edge_type], 'global_edge_index')},
#         global_edge_attr_dict={edge_type: data[edge_type].global_edge_attr
#                                for edge_type in data.edge_types
#                                if hasattr(data[edge_type], 'global_edge_attr')},
#         train_edge_index_dict={edge_type: data[edge_type].train_edge_index
#                                for edge_type in data.edge_types
#                                if hasattr(data[edge_type], 'train_edge_index')},
#         test_edge_index_dict={edge_type: data[edge_type].test_edge_index
#                               for edge_type in data.edge_types
#                               if hasattr(data[edge_type], 'test_edge_index')},
#         add_edge_pool_dict={
#             edge_type: torch.cat([
#                 data[edge_type].train_edge_index,
#                 data[edge_type].test_edge_index
#             ], dim=1)
#             for edge_type in data.edge_types
#             if hasattr(data[edge_type], 'train_edge_index') and hasattr(data[edge_type], 'test_edge_index')
#         }
#     )
#
#     # Pre-generate data with negative samples for both train and test
#     num_epochs = config["training"]["epochs"]
#     batch_size = config["training"]["batch_size"]
#     negative_rate = config["model"]["Negative_rate"]
#
#     # data_list = joblib.load(filename=f"{config['dataset']['processed_dir']}data_list_hetero_1000.pkl")
#     data_list = []
#     start_time = time.time()
#     data = create_data_with_neg_samples_parallel(
#         combined_data_base,
#         device,
#         negative_rate,
#         40,
#         # num_workers=config['dataset'].get('ncores', 10)
#         num_workers=1,
#         node_label_dict=node_label_dict
#     )
#     end_time = time.time()
#     print(f"代码执行耗时: {end_time - start_time:.2f} 秒")
#     data_list.extend(data)
#     data_list = data_list[0:400]
#     joblib.dump(data_list, filename=f"{config['dataset']['processed_dir']}data_list_hetero_1000.pkl")
