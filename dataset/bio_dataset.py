import os
from importlib.metadata import metadata

import joblib
from tqdm import tqdm
from collections import Counter

import yaml
import torch
import torch.nn.functional as F
import logging
import numpy as np
import pandas as pd
from typing import Union, List, Tuple,Dict, Any
from torch_geometric.data import InMemoryDataset, HeteroData, DataLoader, Dataset, Data
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score, average_precision_score


from utils import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Force reconfiguration of logging
)

class BioDataset(InMemoryDataset):
    def __init__(self, config):
        self.config = config
        logging.getLogger().setLevel(logging.INFO)

        ################################
        #    Get paths and parameters from configuration file    #
        ################################
        # Root directory of the dataset
        self.data_root = self.config['data_root']
        # Number of subgraphs
        self.num_graphs = self.config['num_graphs']
        # Whether to use node names
        self.use_node_name = self.config['use_node_name']
        # Whether to read graph labels
        self.use_graph_labels = self.config.get('use_graph_labels', False)

        self.orige_file = self.config.get('orige_file', None)

        # Node label column name
        self.node_label_column = self.config.get('node_label_column', None)
        # Edge timestamp column name
        self.edge_timestamp_column = self.config.get('edge_timestamp_column', None)
        # Edge type column name
        self.edge_type_column = self.config.get('edge_type_column', None)
        self.edge_feature_column = self.config.get('edge_feature_column', None)

        # Edge label column name
        self.edge_label_column = self.config.get('edge_label_column', None)

        # Whether to merge graphs
        self.merge_graphs = self.config.get('merge_graphs', False)

        self.transform = None
        self.pre_transform = None

        # Call parent class constructor
        super(BioDataset, self).__init__(self.data_root, self.transform, self.pre_transform, force_reload=self.config["force_reload"])
        self.data, self.slices = self._load_processed_data()

        self.homogeneous_data = None
        self.node_type_mapping = None
        self.edge_type_mapping = None

    def _load_processed_data(self):
        """
        Load processed data
        :return:
        """
        if self.merge_graphs:
            # Merge subgraphs
            loaded_data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[0]), weights_only=False)
            if isinstance(loaded_data, tuple):
                return loaded_data  # Return (dataset, slices)
            else:
                return loaded_data, None  # Single subgraph case
        else:
            # Non-merge mode
            # Controlled by __getitem__ for loading
            return None, None

    @property
    def raw_dir(self) -> str:
        """
        Read configuration to set raw file directory
        :return:
        """
        return self.config['raw_dir']

    @property
    def raw_file_names(self):
        """
        Read configuration to get raw file list
        :return:
        """
        raw_file_list = []
        for i in range(self.num_graphs):
            if os.path.exists(os.path.join(self.raw_dir, f"graph_{i}_nodes.csv")):
                raw_file_list.append(f"graph_{i}_nodes.csv")
            if os.path.exists(os.path.join(self.raw_dir, f"graph_{i}_edges.csv")):
                raw_file_list.append(f"graph_{i}_edges.csv")
            if os.path.exists(os.path.join(self.raw_dir, f"graph_{i}_node_names.csv")):
                raw_file_list.append(f"graph_{i}_node_names.csv")

        return raw_file_list

    @property
    def processed_dir(self) -> str:
        """
        Read configuration to set processed file directory
        :return:
        """
        return self.config['processed_dir']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        """
        Generate processed file name list
        :return:
        """
        if self.merge_graphs:
            return ['merged_data.pt']
        else:
            return [f'graph_{i}.pt' for i in range(self.num_graphs)
            ]

    @property
    def processed_paths(self) -> List[str]:
        """Return processed file paths."""
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]

    def download(self) -> None:
        pass

    def process(self):

        seed = self.config.get('seed', 42)
        rate = self.config.get('rate', 0.2)

        data_list = []

        # If using labels, read labels from label file
        if self.use_graph_labels:
            labels_df = pd.read_csv(os.path.join(self.raw_dir, 'graph_labels.csv'))

        for i in range(self.num_graphs):
            # Read CSV files from custom raw data directory
            nodes_file = os.path.join(self.raw_dir, f'graph_{i}_nodes.csv')
            edges_file = os.path.join(self.raw_dir, f'graph_{i}_edges.csv')

            # Read raw node and edge data
            nodes = pd.read_csv(nodes_file)
            edges = pd.read_csv(edges_file)

            if self.orige_file:
                orige_df = pd.read_csv(os.path.join(self.raw_dir, self.orige_file))

            # Determine whether to use node name to index each node
            if self.use_node_name:
                # Use node names

                # Read node names
                node_names_df = pd.read_csv(os.path.join(self.raw_dir, f'graph_{i}_node_names.csv'))
                # Initialize node name dictionary
                node_dict = dict()
                # Initialize node id dictionary
                id_dict = dict()
                # Iterate through node names
                for index, row in node_names_df.iterrows():
                    # Check if current node type appears for the first time
                    if row.get("type", None) in id_dict:
                        # If not first occurrence, increment id of this node type by 1
                        id = id_dict[row.get("type", None)] + 1
                    else:
                        # First occurrence of this node type, id is 0
                        id = 0
                    id_dict[row.get("type", None)] = id

                    # Write to node name dictionary
                    node_dict[row["name"]] = {
                        "id": id,
                        "type": row.get("type", None),
                        "cluster":row.get('cluster', -1)
                    }

                # Write node id and type to raw node data
                nodes['_id'] = nodes.iloc[:, 0].map(lambda x: node_dict[x]["id"])
                nodes['_type'] = nodes.iloc[:, 0].map(lambda x: node_dict[x]["type"])
                nodes['_cluster'] = nodes.iloc[:, 0].map(lambda  x: node_dict[x]['cluster'])

                # Create HeteroData object, include labels if needed
                data = HeteroData()

                # Group raw node data by node type
                # Iterate through all groups
                grouped = nodes.groupby('_type')
                for type_name, group in grouped:
                    # Remove node type
                    group = group.drop(columns=['_type'])
                    # Sort by node id
                    group = group.sort_values(by='_id')
                    # Store cluster labels (as integer tensor)
                    if '_cluster' in group.columns:
                        data[str(type_name)].cluster = torch.tensor(group['_cluster'].values, dtype=torch.long)
                        group = group.drop(columns=['_cluster'])
                    else:
                        # If no cluster label, fill with -1
                        data[str(type_name)].cluster = torch.full((len(group),), -1, dtype=torch.long)


                    # Check if node label column is set
                    if self.node_label_column:
                        # If node label exists, extract and remove the column
                        data[str(type_name)].y = torch.tensor(group[self.node_label_column], dtype=torch.float)
                        group = group.drop(columns=[self.node_label_column])

                    if self.orige_file and str(type_name) in ['TF', 'Target']:
                        # Create mapping between node names and IDs
                        name_to_id = pd.Series(group['_id'].values, index=group.iloc[:, 0])

                        # Get rows in orige features matching current group nodes
                        matching_features = orige_df[orige_df.iloc[:, 0].isin(name_to_id.index)]

                        # Add corresponding node IDs to matching_features
                        matching_features['_id'] = matching_features.iloc[:, 0].map(name_to_id)

                        # Sort by node ID
                        matching_features = matching_features.sort_values('_id')

                        # Verify all nodes have corresponding features
                        if len(matching_features) != len(group):
                            missing_nodes = set(group.iloc[:, 0]) - set(matching_features.iloc[:, 0])
                            raise ValueError(f"Missing features for nodes in {type_name}: {missing_nodes}")

                        # Verify ID order consistency
                        if not all(matching_features['_id'].values == group['_id'].values):
                            raise ValueError(f"ID mismatch detected for {type_name} nodes")

                        # Extract features (exclude first column (node name) and _id column)
                        z = torch.tensor(matching_features.iloc[:, 1:-1].values, dtype=torch.float)
                        data[str(type_name)].z = z

                        logging.info(f"Added expression features for {type_name}: shape {z.shape}")

                    # Remove node id
                    group = group.drop(columns=['_id'])

                    # Extract remaining columns as node features
                    data[str(type_name)].x = torch.tensor(group.iloc[:, 1:].values, dtype=torch.float)
                    data[str(type_name)].label = group.iloc[:, 0].values


                # Extract edge indices and edge features
                edge_dict = dict()

                # Record reverse edges to be added
                reverse_edge_dict = dict()

                # Initialize edge type column as None
                edge_type_column = None
                # Check if edge type column is set
                if self.edge_type_column:
                    # If set, extract and remove edge type column from raw edge data
                    edge_type_column = self.edge_type_column
                    edges = edges.drop(columns=[edge_type_column])
                # num = 0
                edge_dict = {}
                edges_df = edges.copy()
                src_col = edges_df.columns[0]
                dst_col = edges_df.columns[1]

                vaild_mask = edges_df[src_col].isin(node_dict.keys()) & edges_df[dst_col].isin(node_dict.keys())
                valid_edges = edges_df[vaild_mask]

                if self.edge_feature_column:
                    feature_cols = [col.strip() for col in self.edge_feature_column.split(',')]
                    # Check if feature columns exist
                    for col in feature_cols:
                        if col not in valid_edges.columns:
                            logging.warning(f"Edge feature column '{col}' not found in edges file")
                    # Keep only existing feature columns
                    valid_feature_cols = [col for col in feature_cols if col in valid_edges.columns]
                    if valid_feature_cols:
                        valid_edges = valid_edges[[src_col, dst_col] + valid_feature_cols]
                    else:
                        valid_edges = valid_edges[[src_col, dst_col]]

                batch_size = 1000
                batchs = [valid_edges[i:i+batch_size] for i in range(0, len(valid_edges), batch_size)]

                with ThreadPoolExecutor(max_workers=self.config.get('ncores',1)) as executor:
                    future_to_batch = {
                        executor.submit(
                            self.process_edge_batch,
                            batch,
                            node_dict,
                            self.config['reverse']
                        ): batch for batch in batchs
                    }

                    for future in tqdm(
                        future_to_batch,
                        total=len(batchs),
                        desc="Processing edge batches"
                    ):
                        try:
                            batch_result = future.result()
                            for edge_type, edges in batch_result.items():
                                if edge_type not in edge_dict:
                                    edge_dict[edge_type] = {'index':[], 'attr':[]}
                                edge_dict[edge_type]['index'].extend(edges['index'])
                                if edges['attr']:
                                    edge_dict[edge_type]['attr'].extend(edges['attr'])
                        except Exception as e:
                            print(f"Error processing batch: {e}")
                            continue

                # Sort ids in order
                type_to_features = {}
                for type_name, group in grouped:
                    # Group is already sorted by _id
                    group = group.sort_values(by='_id')
                    features = group.drop(columns=['_id', '_type', '_cluster']).iloc[:, 1:].values  # Skip first column (name)
                    if len(features) > 0:
                        type_to_features[type_name] = features

                threshold = self.config.get('cor_threshold', None)
                threshold_dict = dict()
                for edge_type in edge_dict.keys():
                    src_type, _, dst_type = edge_type
                    threshold_dict[f"{src_type}_{dst_type}"] = self.config.get(f"{src_type}_{dst_type}",threshold)

                edge_global_dict = self.calculate_node_correlations(
                    type_to_features,
                    edge_dict
                )

                edge_dict = self.deduplicate_edge_dict(edge_dict)
                edge_global_dict = self.deduplicate_edge_dict(edge_global_dict)

                # edge_dict = self.merge_edges(edge_dict)
                # edge_global_dict = self.merge_edges(edge_global_dict)

                # Iterate through generated edge dictionary to populate HeteroData object
                for key in edge_dict:
                    # Get edge indices and transpose to required shape
                    edge_index = np.array(edge_dict[key]["index"]).T
                    # Set edge indices
                    data[key].edge_index = torch.tensor(edge_index, dtype=torch.long)
                    # Check if edge features exist
                    if len(edge_dict[key]["attr"]) > 0:
                        # If exist, set edge features
                        data[key].edge_attr = torch.tensor(np.array(edge_dict[key]["attr"],dtype=float), dtype=torch.float)

                # Add global edges
                for key, edges in edge_global_dict.items():
                    edge_index = np.array(edges["index"]).T
                    data[key].global_edge_index = torch.tensor(edge_index, dtype=torch.long)
                    data[key].global_edge_attr = torch.tensor(np.array(edges["attr"], dtype=float),
                                                             dtype=torch.float)

            else:
                # Do not use node names
                # Todo
                # To be implemented
                pass

            # Get graph labels (if using labels)
            if self.use_graph_labels:
                data["global"].y = torch.tensor([labels_df.loc[labels_df['graph_id'] == i, 'label'].values[0]], dtype=torch.long)

            data_list.append(data)

        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)

        if self.merge_graphs:
            if len(data_list) > 0:
                # Merge all Data objects and save as one file
                if len(data_list) == 1:
                    # Single subgraph case, do not use collate()
                    torch.save(data_list[0], os.path.join(self.processed_dir, self.processed_file_names[0]))
                    logging.info(f"Single subgraph saved to {self.processed_file_names[0]}")
                else:
                    data, slices = self.collate(data_list)
                    torch.save((data, slices), os.path.join(self.processed_dir, self.processed_file_names[0]))
                    logging.info(f"All graph data merged and saved to {self.processed_file_names[0]}")
            else:
                raise Exception("The dataset is empty.")
        else:
            # Save each Data object separately
            for i, data in enumerate(data_list):
                torch.save((data, None), self.processed_file_names[i])
                logging.info(f"Graph data {i} saved to {self.processed_file_names[i]}")

    def __getitem__(self, idx):
        """
        Load processed files according to configuration
        :param idx:
        :return:
        """
        if self.merge_graphs:
            if self.slices is None:
                # Single subgraph, return directly
                return self._data
            else:
                # Multiple subgraphs, extract using slice information
                return self.get(idx)
        else:
            data, _ = torch.load(self.processed_file_names[idx])
            return data

    def __len__(self):
        """
        Return the size of the dataset.
        :return:
        """
        if self.merge_graphs:
            # Load data and slice information from merged file
            if self.slices is None:
                return 1  # Single subgraph case
            else:
                # Return number of merged graphs, determined by len(slices['x']) - 1
                return len(self.slices['x']) - 1
        else:
            # If not merged, return number of graphs directly
            return self.num_graphs

    def get(self, idx):
        """Extract subgraph from merged data by index."""
        data = HeteroData()

        # Iterate through each attribute and extract data according to slice information
        for key in self._data.keys():
            item = self._data[key]
            if torch.is_tensor(item):
                start, end = self.slices[key][idx], self.slices[key][idx + 1]
                data[key] = item[start:end]
            else:
                data[key] = item

        return data

    def get_metadata(self):
        dataset_metadata = {
            'node_types': self.data.node_types,
            'edge_types': self.data.edge_types,
        }

        return dataset_metadata

    def to_homegeneous(self):

        if self.homogeneous_data is not None:
            return self.homogeneous_data

        data = self.data
        if self.node_type_mapping is None:
            self.node_type_mapping = {node_type:idx for idx, node_type in enumerate(data.node_types)}
        if self.edge_type_mapping is None:
            self.edge_type_mapping = {edge_type:idx for idx, edge_type in enumerate(data.edge_types)}

        x_list = []
        edge_index_list = []
        edge_attr_list = []
        node_type_list = []
        edge_type_list = []

        cumsum_nodes = {data.node_types[0]: 0}
        for i in range(1, len(data.node_types)):
            cumsum_nodes[data.node_types[i]] = cumsum_nodes[data.node_types[i-1]] + \
                data[data.node_types[i-1]].x.size(0)

        for node_type in data.node_types:
            num_nodes = data[node_type].x.size(0)
            x_list.append(data[node_type].x)

            node_type_list.append(torch.full((num_nodes, ),
                                             self.node_type_mapping[node_type],
                                             dtype=torch.long))


        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            edge_index = data[edge_type].edge_index

            edge_index_offset = edge_index.clone()
            edge_index_offset[0]+=cumsum_nodes[src_type]
            edge_index_offset[1]+=cumsum_nodes[dst_type]

            edge_index_list.append(edge_index_offset)
            edge_type_list.append(torch.full((edge_index.size(1),),
                                             self.edge_type_mapping[edge_type],
                                             dtype=torch.long))

            if hasattr(data[edge_type], 'edge_attr'):
                edge_attr_list.append(data[edge_type].edge_attr)

        x = torch.cat(x_list, dim=0)
        edge_index = torch.cat(edge_index_list, dim=1)
        node_type = torch.cat(node_type_list)
        edge_type = torch.cat(edge_type_list)

        homogeneous_data = Data(
            x=x,
            edge_index=edge_index,
            node_type=node_type,
            edge_type=edge_type
        )

        if edge_attr_list:
            homogeneous_data.edge_attr = torch.cat(edge_attr_list, dim=0)

        homogeneous_data.num_node_types = len(data.node_types)
        homogeneous_data.num_edge_types = len(data.edge_types)
        homogeneous_data.node_type_dict = self.node_type_mapping
        homogeneous_data.edge_type_dict = self.edge_type_mapping

        self.homogeneous_data = homogeneous_data

        return homogeneous_data

    def process_edge_batch(self, batch_df: pd.DataFrame, node_dict: Dict[str, Dict[str, Any]],
                           reverse=True) -> Dict:
        """
        Process a batch of edges to create edge dictionary with indices and attributes

        Args:
            batch_df (pd.DataFrame): Batch of edges to process
            node_dict (Dict): Dictionary containing node information

        Returns:
            Dict: Dictionary containing processed edges
        """
        local_edge_dict = {}
        src_col = batch_df.columns[0]
        dst_col = batch_df.columns[1]

        src_types = batch_df[src_col].map(lambda x: node_dict[x]['type'])
        dst_types = batch_df[dst_col].map(lambda x: node_dict[x]['type'])

        src_ids = batch_df[src_col].map(lambda x: node_dict[x]['id'])
        dst_ids = batch_df[dst_col].map(lambda x: node_dict[x]['id'])

        if self.edge_type_column:
            edge_types = list(zip(src_types, batch_df[self.edge_type_column], dst_types))
            reverse_types = list(zip(dst_types, batch_df[self.edge_type_column], src_types))
        else:
            edge_types = list(zip(src_types, ['edge'] * len(batch_df), dst_types))
            reverse_types = list(zip(dst_types, ['edge'] * len(batch_df), src_types))

        # Use set to store processed edges
        processed_edges = {edge_type: set() for edge_type in set(edge_types + reverse_types)}

        for idx, (edge_type, reverse_type) in enumerate(zip(edge_types, reverse_types)):
            # Convert to tuple for consistent comparison
            edge = (int(src_ids.iloc[idx]), int(dst_ids.iloc[idx]))
            reverse_edge = (int(dst_ids.iloc[idx]), int(src_ids.iloc[idx]))

            # Initialize dictionaries for edge types if needed
            if edge_type not in local_edge_dict:
                local_edge_dict[edge_type] = {"index": [], "attr": []}

            # Process forward edge if not already processed
            if edge not in processed_edges[edge_type]:
                local_edge_dict[edge_type]["index"].append(list(edge))
                processed_edges[edge_type].add(edge)
                if len(batch_df.columns) > 2:
                    local_edge_dict[edge_type]["attr"].append(batch_df.iloc[idx, 2:].values)

            if reverse:
                if reverse_type not in local_edge_dict:
                    local_edge_dict[reverse_type] = {"index": [], "attr": []}
                # Process reverse edge if not already processed
                if reverse_edge not in processed_edges[reverse_type]:
                    local_edge_dict[reverse_type]["index"].append(list(reverse_edge))
                    processed_edges[reverse_type].add(reverse_edge)
                    if len(batch_df.columns) > 2:
                        local_edge_dict[reverse_type]["attr"].append(batch_df.iloc[idx, 2:].values)

        return local_edge_dict

    def merge_edges(self, edge_dict):
        """
        Merge forward and reverse edges in the edge dictionary.

        Args:
            edge_dict (dict): Dictionary containing edges to be merged

        Returns:
            dict: Dictionary containing merged edges
        """
        processed_edge_types = set()
        merged_edge_dict = {}

        for edge_type in list(edge_dict.keys()):
            src_type, rel, dst_type = edge_type

            # Skip if same node type or already processed
            if src_type == dst_type or (src_type, dst_type) in processed_edge_types:
                continue

            # Get forward and reverse edge types
            forward_type = (src_type, rel, dst_type)
            reverse_type = (dst_type, rel, src_type)

            # Process forward edges
            forward_edges = torch.tensor(np.array(edge_dict[forward_type]['index']).T)
            forward_attrs = torch.tensor(np.array(edge_dict[forward_type]['attr'], dtype=float),dtype=torch.float) if edge_dict[forward_type][
                'attr'] else None

            # If reverse edges exist, merge them
            if reverse_type in edge_dict:
                reverse_edges = torch.tensor(np.array(edge_dict[reverse_type]['index']).T)
                reverse_attrs = torch.tensor(np.array(edge_dict[reverse_type]['attr'], dtype=float),dtype=torch.float) if edge_dict[reverse_type][
                    'attr'] else None

                # Merge edges
                merged_edges = torch.cat([forward_edges, reverse_edges], dim=1)
                merged_edge_dict[forward_type] = {
                    'index': merged_edges.t().tolist(),
                    'attr': torch.cat([forward_attrs,
                                       reverse_attrs]).tolist() if forward_attrs is not None and reverse_attrs is not None else []
                }
            else:
                merged_edge_dict[forward_type] = edge_dict[forward_type]

            # Mark edge types as processed
            processed_edge_types.add((src_type, dst_type))
            processed_edge_types.add((dst_type, src_type))

        return merged_edge_dict

    def deduplicate_edge_dict(self, edge_dict):
        """
        Remove duplicate edges from edge_dict for each edge type.
        Duplicates are edges with the same source and destination nodes.

        Args:
            edge_dict (dict): Original edge dictionary with format:
                {edge_type: {'index': [[src, dst], ...], 'attr': [attr1, ...]}}

        Returns:
            dict: Deduplicated edge dictionary with the same format
        """
        deduped_dict = {}

        for edge_type, edges in edge_dict.items():
            # Track seen edges using a set for O(1) lookup
            seen_edges = set()
            deduped_indices = []
            deduped_attrs = []

            # Process each edge and its attributes
            for idx, edge in enumerate(edges['index']):
                # Convert edge to tuple for hashability
                edge_tuple = tuple(edge)

                # Only keep first occurrence of each edge
                if edge_tuple not in seen_edges:
                    seen_edges.add(edge_tuple)
                    deduped_indices.append(edge)

                    # Keep corresponding attributes if they exist
                    if edges['attr']:
                        deduped_attrs.append(edges['attr'][idx])

            # Store deduplicated edges and attributes
            deduped_dict[edge_type] = {
                'index': deduped_indices,
                'attr': deduped_attrs if edges['attr'] else []
            }

            # Log deduplication results
            logging.info(f"Edge type {edge_type}: Original edges: {len(edges['index'])}, "
                         f"After deduplication: {len(deduped_indices)}, "
                         f"Removed {len(edges['index']) - len(deduped_indices)} duplicates")

        return deduped_dict

    def calculate_node_correlations(self, type_to_features, edge_dict):
        """
        Calculate correlations between nodes using pre-grouped data and existing edge types

        Args:
            type_to_features: Dictionary mapping node types to their features
            edge_dict: Dictionary containing existing edges and their types

        Returns:
            dict: Global edge dictionary containing correlation-based edges
        """
        edge_global_dict = {k: {"index": [], "attr": []} for k in edge_dict.keys()}
        cor_method = self.config.get('global_network_method', None)
        threshold_m = self.config.get('threshold_m', None)
        threshold = self.config.get('cor_threshold', None)
        threshold_dict = {}

        # Build threshold dictionary
        for edge_type in edge_dict.keys():
            src_type, _, dst_type = edge_type
            threshold_dict[f"{src_type}_{dst_type}"] = self.config.get(f"{src_type}_{dst_type}", threshold)

        for edge_type in edge_dict.keys():
            src_type, _, dst_type = edge_type
            features_A = type_to_features[src_type]
            features_B = type_to_features[dst_type]

            if cor_method == 'pearson':
                n = features_A.shape[1]
                features_A = (features_A - features_A.mean(axis=1)[:, None]) / (features_A.std(axis=1)[:, None] + 1e-10)
                features_B = (features_B - features_B.mean(axis=1)[:, None]) / (features_B.std(axis=1)[:, None] + 1e-10)

                corr_matrix = np.dot(features_A, features_B.T) / n

                if src_type == dst_type:
                    np.fill_diagonal(corr_matrix, 0)

                if threshold_m == 'percent':
                    flat_corr = np.abs(corr_matrix.flatten())
                    threshold_ = np.sort(flat_corr)[::-1][
                        int(len(flat_corr) * threshold_dict[f"{src_type}_{dst_type}"])]
                    logging.info(f"{src_type}_{dst_type} threshold: {threshold_}")

                high_corr_indics = np.where(np.abs(corr_matrix) > threshold_)
                rows, cols = high_corr_indics

                if src_type == dst_type:
                    for i, j in zip(rows, cols):
                        edge_global_dict[edge_type]['index'].append([int(i), int(j)])
                        edge_global_dict[edge_type]['attr'].append([float(corr_matrix[i, j])])

                        edge_global_dict[edge_type]['index'].append([int(j), int(i)])
                        edge_global_dict[edge_type]['attr'].append([float(corr_matrix[j, i])])
                else:
                    for i, j in zip(*high_corr_indics):
                        edge_global_dict[edge_type]['index'].append([int(i), int(j)])
                        edge_global_dict[edge_type]['attr'].append([float(corr_matrix[i, j])])

        return edge_global_dict


class NegativeSamplesDataset(Dataset):
    def __init__(self, base_data, neg_samples):
        self.base_data = base_data
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.neg_samples)

    def __getitem__(self, idx):
        train_neg, test_neg = self.neg_samples[idx]
        return train_neg, test_neg


def create_data_loader(
        config: Dict[str, Any],
        device,
        data=None,
        fold_splits=None,
        neg_sample_path: str=None
) -> DataLoader:
    """
    Package all data to generate data_loader
    """

    gpu_x_dict = {node_type: tensor.to(device) for node_type, tensor in data.x_dict.items()}
    gpu_z_dict = {}
    for node_type in ['TF', 'Target']:
        if hasattr(data[node_type], 'z'):
            gpu_z_dict[node_type] = data[node_type].z.to(device)
    gpu_prior_edge_dict = {}
    gpu_prior_edge_attr_dict = {}
    gpu_global_edge_dict = {}
    gpu_global_edge_attr_dict = {}
    gpu_train_edge_dict = {}
    gpu_test_edge_dict = {}

    for edge_type in data.edge_types:
        # Add prior edges
        if hasattr(data[edge_type], 'edge_index'):
            gpu_prior_edge_dict[edge_type] = data[edge_type].edge_index.to(device)
            if hasattr(data[edge_type], 'edge_attr'):
                gpu_prior_edge_attr_dict[edge_type] = data[edge_type].edge_attr.to(device)

        # Process global edges (co-expression relationships)
        if hasattr(data[edge_type], 'global_edge_index'):
            gpu_global_edge_dict[edge_type] = data[edge_type].global_edge_index.to(device)
            if hasattr(data[edge_type], 'global_edge_attr'):
                gpu_global_edge_attr_dict[edge_type] = data[edge_type].global_edge_attr.to(device)

        # Process training edges
        if hasattr(data[edge_type], 'train_edge_index'):
            gpu_train_edge_dict[edge_type] = data[edge_type].train_edge_index.to(device)

        # Process test edges
        if hasattr(data[edge_type], 'test_edge_index'):
            gpu_test_edge_dict[edge_type] = data[edge_type].test_edge_index.to(device)

    gpu_regression_train_masks = {}
    gpu_regression_test_masks = {}
    for node_type in data.node_types:
        if node_type in fold_splits:
            gpu_regression_train_masks[node_type] = fold_splits[node_type]['train_mask'].to(device)
            gpu_regression_test_masks[node_type] = fold_splits[node_type]['test_mask'].to(device)

    data = Data(
        x_dict=gpu_x_dict,
        z_dict=gpu_z_dict,
        prior_edge_index_dict=gpu_prior_edge_dict,
        prior_edge_attr_dict=gpu_prior_edge_attr_dict,
        global_edge_index_dict=gpu_global_edge_dict,
        global_edge_attr_dict=gpu_global_edge_attr_dict,
        train_edge_index_dict=gpu_train_edge_dict,
        test_edge_index_dict=gpu_test_edge_dict,
        regression_train_mask_dict=gpu_regression_train_masks,
        regression_test_mask_dict=gpu_regression_test_masks
    )

    batch_size = config["training"]["batch_size"]

    # Read negative samples
    neg_samples = joblib.load(neg_sample_path)

    dataset = NegativeSamplesDataset(data, neg_samples)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True if device.type == 'cuda' else False
    )

    return data_loader


class NetworkBuider:
    """
    The NetworkBuider is a utility class in the SMOGT framework for constructing a heterogeneous regulatory network (HRNet) from node embeddings.
    Based on user-specified edge types (e.g., TF‑TF, TF‑CRE, CRE‑CRE, CRE‑Target, etc.),
    it computes similarity scores between node embeddings (using Pearson correlation coefficient, dot product, or cosine similarity),
    filters high-confidence edges, and saves the resulting network as files in the multiplex and bipartite subdirectories,
    which can then be used by downstream modules such as MRWR for driver regulator identification and perturbation simulation.

    Args:
        config (dict):
            A dictionary containing information such as data paths and model configuration,
            typically loaded from a YAML configuration file.
            It should at least contain a 'dataset' sub-configuration for constructing the BioDataset.

        edge_types (List[Tuple[str, str]], optional):
            A list of edge types to be constructed, where each element is a tuple (source node type, target node type).
            For example: [('TF','TF'), ('TF','CRE'), ('CRE','CRE'), ('CRE','Target')].
            If None, edge types are automatically inferred based on the threshold or topK parameters
            (but explicit specification is recommended).

        threshold (Dict[str, float], optional):
            A dictionary for filtering edges by similarity threshold, where keys are in the format "source_target"
            (e.g., "TF_CRE") and values are thresholds (0–1).
            When this parameter is provided, topK and topK_type are ignored.

        topK (int, optional):
            If using top‑K filtering, specifies the number of edges to retain (global or local).
            This parameter takes effect when threshold is None.

        topK_type (str, optional):
            Top‑K filtering mode: 'globel' (retain the top K edges from the entire similarity matrix) or
            'local' (retain the top K highest-scoring edges for each source node).

        topp_CRE (float, optional):
            A special proportion parameter used only for constructing CRE‑CRE edges.
            Due to the large number of CREs and the biological constraint that interactions typically occur
            within the same chromosome, this parameter controls the proportion of edges retained from
            intra-chromosomal candidate edges (e.g., 0.01 means retaining the top 1% highest-scoring edges
            within each chromosome).

        method (str, optional):
            The method used to compute node pair similarity.
            Supports 'pearson' (Pearson correlation coefficient), 'dot' (dot product followed by sigmoid),
            and 'cosine' (cosine similarity).
    """
    def __init__(self,
                 config,
                 edge_types: List=None,
                 threshold: Dict=None,
                 topK: int=None,
                 topK_type: str='globel',
                 topp_CRE: float=0.1,
                 method: str='pearson'):
        self.config = config
        self.embedding = None
        self.node_data = None
        self.node_groups = {}
        self.networks = {}
        self.node_types = []
        self.edge_types = edge_types
        self.threshold = threshold
        self.topK = topK
        self.topK_type = topK_type
        self.topp_CRE=topp_CRE
        self.method=method

    def load_data(self, file_path) -> None:
        self.embedding = self._load_embedding_data(file_path)
        self.node_data = self._load_node_classification()
        self._group_nodes()

    def _load_embedding_data(self, file_path: str) -> pd.DataFrame:
        embedding_dim = self.config['model']['embedding_dim']
        df = pd.read_csv(file_path)
        dim_cols = [f'dim_{i}' for i in range(embedding_dim)]

        if 'gene_id' in df.columns:
            select_cols = dim_cols+['gene_id']
        else:
            select_cols = dim_cols+[df.columns[-1]]

        return df[select_cols]

    def _load_node_classification(self) -> pd.DataFrame:
        node_path = os.path.join(self.config['dataset']['raw_dir'], 'graph_0_node_names.csv')
        return pd.read_csv(node_path)

    def _group_nodes(self) -> None:
        self.node_types = self.node_data['type'].value_counts().index.tolist()
        for node_type in self.node_types:
            nodes = self.node_data[self.node_data['type']==node_type]
            self.node_groups[node_type] = self.embedding[self.embedding['gene_id'].isin(nodes['name'])]

    def _compute_similarity_matrix(self,
                                   feature_A: pd.DataFrame,
                                   feature_B: pd.DataFrame=None,
                                   method: str='pearson') -> torch.Tensor:
        feature_A = torch.tensor(feature_A.iloc[:, :-1].values, dtype=torch.float)

        if feature_B is None:
            feature_B = feature_A
        else:
            feature_B = torch.tensor(feature_B.iloc[:, :-1].values, dtype=torch.float)

        if method=='pearson':
            correlation = self._compute_pearson(feature_A, feature_B)
        elif method=='dot':
            correlation = self._compute_dot(feature_A, feature_B)
        elif method=='cosine':
            correlation = self._compute_cosine(feature_A, feature_B)

        if torch.equal(feature_A, feature_B):
            correlation.fill_diagonal_(0)

        return correlation


    def _compute_pearson(self,
                         feature_A: torch.Tensor,
                         feature_B: torch.Tensor) -> torch.Tensor:
        feature_A = feature_A-feature_A.mean(dim=1, keepdim=True)
        feature_B = feature_B-feature_B.mean(dim=1, keepdim=True)

        std_A = torch.sqrt(torch.sum(feature_A**2, dim=1, keepdim=True))
        std_B = torch.sqrt(torch.sum(feature_B**2, dim=1, keepdim=True))

        std_A = torch.clamp(std_A, min=1e-8)
        std_B = torch.clamp(std_B, min=1e-8)

        feature_A = feature_A/std_A
        feature_B = feature_B/std_B

        n = feature_A.size(1)
        correlation = torch.mm(feature_A, feature_B.t())/n

        correlation = torch.clamp(correlation, min=-1, max=1)

        return correlation

    def _compute_dot(self,
                     feature_A: torch.Tensor,
                     feature_B: torch.Tensor
                      ) -> torch.Tensor:
        # feature_A = F.normalize(feature_A, p=2, dim=1)
        # feature_B = F.normalize(feature_B, p=2, dim=1)
        correlation = torch.mm(feature_A, feature_B.t())
        return torch.sigmoid(correlation)

    def _compute_cosine(self,
                     feature_A: torch.Tensor,
                     feature_B: torch.Tensor
                      ) -> torch.Tensor:
        norm_A = torch.sqrt(torch.sum(feature_A**2, dim=1, keepdim=True))
        norm_B = torch.sqrt(torch.sum(feature_B ** 2, dim=1, keepdim=True))

        norm_A = torch.clamp(norm_A, min=1e-8)
        norm_B = torch.clamp(norm_B, min=1e-8)

        feature_A = feature_A/norm_A
        feature_B = feature_B/norm_B

        return torch.mm(feature_A, feature_B.t())


    def _select_edges_by_topk(self,
                              similarity_matrix: torch.Tensor,
                              k: int,
                              method: str='globel') -> Tuple[torch.tensor, torch.Tensor]:

        if method=='globel':
            values, indices = torch.topk(similarity_matrix.view(-1), k)
            rows = indices // similarity_matrix.size(1)
            cols = indices % similarity_matrix.size(1)
            edges = torch.stack([rows, cols])
        else:
            values, indices = torch.topk(similarity_matrix, k, dim=1)
            rows = torch.arange(similarity_matrix.size(0)).view(-1, 1).expand(-1, k).flatten()
            edges = torch.stack([rows, indices.flatten()])
            values = values.flatten()

        return edges, values


    def _select_edges_by_threshold(self,
                                   similarity_matrix: torch.Tensor,
                                   threshold: float
                                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = similarity_matrix>=threshold
        edges = torch.nonzero(mask).t()
        values = similarity_matrix[edges[0], edges[1]]

        return edges, values

    def buide_networks(self) -> None:
        for src, dst in self.edge_types:
            if src not in self.node_types or dst not in self.node_types:
                continue

            # Process CRE by chromosome
            if src=='CRE' and dst=='CRE':
                cre_data = self.node_groups['CRE'].copy()
                cre_data['chromosome'] = cre_data['gene_id'].apply(lambda x: x.split('-')[0])
                chromosomes = cre_data['chromosome'].unique()

                all_edges = []

                for chrom in chromosomes:
                    chrom_data = cre_data[cre_data['chromosome'] == chrom]
                    chrom_data = chrom_data.iloc[:, :-1]

                    if len(chrom_data) > 1:
                        similarity_matrix = self._compute_similarity_matrix(
                            chrom_data,
                            None,
                            self.method
                        )

                        total_possible_edges = len(chrom_data) * (len(chrom_data))
                        k = int(total_possible_edges * self.topp_CRE)
                        edges, values = self._select_edges_by_topk(
                            similarity_matrix,
                            k=k,
                            method=self.topK_type
                        )

                        # Get node indices of current chromosome
                        chrom_indices = chrom_data.index.values

                        # Convert edge indices to actual gene IDs
                        network_df = pd.DataFrame({
                            'from': chrom_data['gene_id'].values[edges[0].numpy()],
                            'to': chrom_data['gene_id'].values[edges[1].numpy()],
                            'value': values.numpy()
                        })

                        all_edges.append(network_df)

                # Merge edges from all chromosomes
                if all_edges:
                    self.networks[f"{src}_{dst}"] = pd.concat(all_edges, ignore_index=True)
            else:
                similarity_matrix = self._compute_similarity_matrix(
                    self.node_groups[src],
                    None if src==dst else self.node_groups[dst],
                    self.method
                )

                if self.threshold is not None:
                    threshold = self.threshold[f"{src}_{dst}"]
                    edges, values = self._select_edges_by_threshold(similarity_matrix,
                                                                    threshold)
                else:
                    edges, values = self._select_edges_by_topk(similarity_matrix,
                                                               k=self.topK,
                                                               method=self.topK_type)
                network_df = pd.DataFrame({
                    'from': self.node_groups[src]['gene_id'].values[edges[0].numpy()],
                    'to': self.node_groups[dst]['gene_id'].values[edges[1].numpy()],
                    'value': values.numpy()
                })

                self.networks[f"{src}_{dst}"] = network_df

    def save_networks(self, base_dir: str=None) -> None:
        for subdir in ['multiplex', 'bipartite', 'seed', 'config']:
            os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

        for network_name, network_df in self.networks.items():
            src, dst = network_name.split('_')
            if src==dst:
                network_dir = os.path.join(base_dir, 'multiplex', src)
                os.makedirs(network_dir, exist_ok=True)

                file_path = os.path.join(network_dir, f"{src}.csv")

                save_df = network_df[['from', 'to', 'value']].copy()

                save_df.to_csv(
                    file_path,
                    sep='\t',
                    index=False,
                    header=False
                )
            else:
                network_dir = os.path.join(base_dir, 'bipartite')
                file_path = os.path.join(network_dir, f"{src}_{dst}.csv")

                save_df = network_df[['from', 'to', 'value']].copy()

                save_df.to_csv(
                    file_path,
                    sep='\t',
                    index=False,
                    header=False
                )

class EvaluationBuider:
    """
    The EvaluationBuider class is a core tool in the SMOGT framework for evaluating the quality of node embeddings.
    Based on trained node embeddings (e.g., the z_dict output from the SimplifiedGraphAtten_regression model),
    it quantifies whether the biological relationships learned by the model align with ground truth regulatory networks
    (based on experimental data such as ChIP‑seq or Hi‑C) by calculating prediction accuracy for specified edge types
    (e.g., TF-CRE, CRE-CRE).

    Args:
        config (dict):
            A dictionary containing information such as data paths and model configuration,
            typically loaded from a YAML configuration file.
            It should at least contain a 'dataset' sub-configuration for constructing the BioDataset.

        method (str):
            The method used to compute node pair similarity.
            Supports 'dot' (dot product followed by sigmoid) and 'cosine' (cosine similarity).
            This value will be used to calculate scores between embeddings during evaluation.

        dataset (BioDataset, optional):
            A loaded dataset object containing node types, features, ground truth edge labels, etc.
            If not provided, it will be automatically created based on the config.

        embedding_path (str):
            Path to the CSV file containing node embeddings.
            The file should at least contain a 'gene_id' column and embedding vector columns
            named dim_0, dim_1, etc.

        edge_attr_index (int, default=-1):
            The column index in the dataset's edge attributes (edge_attr) where ground truth labels are stored.
            Labels are typically 1 (positive) or 0 (negative). Default -1 indicates using the last column as the label.

        negative_ratio (float, default=1.0):
            The ratio of negative samples to positive samples. For example, 1.0 means generating an equal number of
            negative samples as positive edges.
    """
    def __init__(self,
                 config: Dict=None,
                 method: str='dot',
                 dataset: BioDataset=None,
                 embedding_path: str=None,
                 edge_attr_index: str=-1,
                 negative_ratio: float=1.0):
        self.config = config
        self.method = method
        self.embedding_df = None
        self.embedding_df_type = None
        self.embedding_path = embedding_path
        self.dataset = dataset
        self.node_to_gene = {}
        self.gene_to_node = {}
        self.edge_attr_index = edge_attr_index
        self.negative_ratio = negative_ratio

        if self.dataset is None:
            self.dataset = BioDataset(config=config['dataset'])

        self._extract_id_map()

    def _extract_id_map(self) -> None:
        for node_type in self.dataset.data.node_types:
            if hasattr(self.dataset.data[node_type], 'label'):
                labels = self.dataset.data[node_type].label
                node_ids = range(len(labels))

                gene_to_node = {geneid:nodeid for nodeid, geneid in zip(node_ids, labels)}
                node_to_gene = {nodeid:geneid for nodeid, geneid in zip(node_ids, labels)}

                self.gene_to_node[node_type] = gene_to_node
                self.node_to_gene[node_type] = node_to_gene

    def load_embedding(self) -> None:
        self.embedding_df = pd.read_csv(self.embedding_path)
        self.embedding_df_type = self._group_embedding()


    def _group_embedding(self) -> Dict[str, Dict]:
        group_embedding = {}

        for node_type in self.dataset.data.node_types:
            node_to_gene = self.node_to_gene[node_type]
            gene_to_node = self.gene_to_node[node_type]

            group_embedding_ = self.embedding_df[self.embedding_df['gene_id'].isin(gene_to_node.keys())]

            if len(group_embedding_)>0:
                dim_cols = [col for col in group_embedding_.columns if col.startswith('dim_')]

                embedding_tensor = torch.tensor(
                    group_embedding_[dim_cols].values,
                    dtype=torch.float
                )

                geneids = group_embedding_['gene_id'].values

                gene_to_index = {geneid:idx for idx, geneid in enumerate(geneids)}

                gene_to_node_ = {geneid:gene_to_node[geneid] for geneid in geneids if geneid in gene_to_node}

                group_embedding[node_type] = {
                    'embedding': embedding_tensor,
                    'geneid':geneids,
                    'gene_to_index':gene_to_index,
                    'gene_to_node':gene_to_node_
                }

        return group_embedding

    def analysis(self,
                 edge_types: List[str]=None) -> Dict:
        """
        This method calculates AUC and AUPR for specified edge types respectively, evaluating the ability of embeddings
        to distinguish between positive and negative edges
        """
        result = {}

        for edge_type_str in edge_types:
            src, dst = edge_type_str.split('-')
            edge_type = (src, 'edge', dst)

            pos_edge_index, pos_edge_attr = self._get_edges(edge_type)

            pos_edges = self._convert_edge_index_to_geneid(pos_edge_index,
                                                           src,
                                                           dst)
            neg_edges = self._generate_negative_samples(
                edge_type,
                pos_edges
            )

            pos_scores = self._calculcate(self.embedding_df_type[src],
                                         self.embedding_df_type[dst],
                                         pos_edges)
            neg_scores = self._calculcate(self.embedding_df_type[src],
                                         self.embedding_df_type[dst],
                                         neg_edges)

            scores = torch.cat([pos_scores, neg_scores]).numpy()
            labels = torch.cat([torch.ones(len(pos_scores)),
                                torch.zeros(len(neg_scores))]).numpy()

            auc = roc_auc_score(labels, scores)
            aupr = average_precision_score(labels, scores)

            result[edge_type_str] = {
                'auc': auc,
                'aupr': aupr
            }

        return result

    def _get_edges(self, edge_type: Tuple[str, str, str]):
        edge_index = self.dataset.data[edge_type].edge_index
        edge_attr = self.dataset.data[edge_type].edge_attr

        pos_edge_mask = edge_attr[:, self.edge_attr_index]==1
        pos_edge_index = edge_index[:, pos_edge_mask]

        pos_edge_attr = edge_attr[pos_edge_mask]

        return (pos_edge_index, pos_edge_attr)

    def _convert_edge_index_to_geneid(self,
                                      edge_index: torch.Tensor,
                                      src: str=None,
                                      dst: str=None) -> Tuple[List[str], List[str]]:
        node_to_gene_src = self.node_to_gene[src]
        node_to_gene_dst = self.node_to_gene[dst]

        src_ids = []
        dst_ids = []

        for i in range(edge_index.size(1)):
            src_node_id = edge_index[0, i].item()
            dst_node_id = edge_index[1, i].item()

            src_ids.append(node_to_gene_src[src_node_id])
            dst_ids.append(node_to_gene_dst[dst_node_id])

        return (src_ids, dst_ids)


    def _generate_negative_samples(self,
                                   edge_type: Tuple[str, str, str],
                                   pos_edges: Tuple[List, List]) -> Tuple[List, List]:
        src, _, dst = edge_type
        num_pos_edges = len(pos_edges[0])
        num_neg_edges = int(num_pos_edges * self.negative_ratio)

        pos_edge_index = set(zip(pos_edges[0], pos_edges[1]))

        src_geneids = list(self.embedding_df_type[src]['geneid'])
        dst_geneids = list(self.embedding_df_type[dst]['geneid'])

        if src=='CRE' and dst=='CRE':
            cre_ids = src_geneids
            chromosomes = np.array([cre_id.split('-')[0] for cre_id in cre_ids])
            unique_chromosomes = np.unique(chromosomes)

            neg_src = []
            neg_dst = []

            for chrom in unique_chromosomes:
                indices = np.where(chromosomes==chrom)[0]

                if len(indices)>1:
                    chrom_neg_count = max(1, int(num_neg_edges * len(indices)/len(cre_ids)))
                    chrom_neg_edges = []

                    while len(chrom_neg_edges)<chrom_neg_count:
                        i = np.random.choice(indices)
                        j = np.random.choice(indices)

                        if i!=j:
                            src_id = cre_ids[i]
                            dst_id = cre_ids[j]

                            if (src_id, dst_id) not in pos_edge_index and (src_id, dst_id) not in chrom_neg_edges:
                                chrom_neg_edges.append((src_id, dst_id))

                    for src, dst in chrom_neg_edges:
                        neg_src.append(src)
                        neg_dst.append(dst)
        elif src=='TF' and dst=='CRE':
            unique_TFs = list(pos_edges[0])
            gene_to_index = {geneid:idx for idx, geneid in enumerate(unique_TFs)}
            index_to_gene = {idx:geneid for idx, geneid in enumerate(unique_TFs)}

            cre_gene_to_index = {geneid:idx for idx, geneid in enumerate(dst_geneids)}
            index_to_cre_gene= {idx:geneid for idx, geneid in enumerate(dst_geneids)}

            local_pos_edges = torch.zeros((2, len(pos_edges[0])), dtype=torch.long)
            for i, (tf_id, cre_id) in enumerate(zip(pos_edges[0], pos_edges[1])):
                print(i)
                local_pos_edges[0, i] = gene_to_index.get(tf_id, 0)
                local_pos_edges[1, i] = cre_gene_to_index.get(cre_id, 0)

            local_neg_edges = negative_sampling(
                edge_index=local_pos_edges,
                num_nodes=(len(unique_TFs), len(dst_geneids)),
                num_neg_samples=num_neg_edges
            )

            neg_src = [index_to_gene[idx.item()] for idx in local_neg_edges[0]]
            neg_dst = [index_to_cre_gene[idx.item()] for idx in local_neg_edges[1]]


        else:
            src_gene_to_index = {gene_id: idx for idx, gene_id in enumerate(src_geneids)}
            src_index_to_gene = {idx: gene_id for gene_id, idx in src_gene_to_index.items()}

            dst_gene_to_index = {gene_id: idx for idx, gene_id in enumerate(dst_geneids)}
            dst_index_to_gene = {idx: gene_id for gene_id, idx in dst_gene_to_index.items()}

            # Convert gene IDs of positive edges to local indices
            local_pos_edges = torch.zeros((2, len(pos_edges[0])), dtype=torch.long)
            for i, (src_id, dst_id) in enumerate(zip(pos_edges[0], pos_edges[1])):
                if src_id in src_gene_to_index and dst_id in dst_gene_to_index:
                    local_pos_edges[0, i] = src_gene_to_index[src_id]
                    local_pos_edges[1, i] = dst_gene_to_index[dst_id]

            # Generate negative samples using negative_sampling
            local_neg_edges = negative_sampling(
                edge_index=local_pos_edges,
                num_nodes=(len(src_geneids), len(dst_geneids)),
                num_neg_samples=num_neg_edges
            )

            # Convert local indices back to gene IDs
            neg_src = [src_index_to_gene[idx.item()] for idx in local_neg_edges[0]]
            neg_dst = [dst_index_to_gene[idx.item()] for idx in local_neg_edges[1]]

        return (neg_src, neg_dst)

    def _calculcate(self,
                    src_embedding: Dict=None,
                    dst_embedding: Dict=None,
                    edge_index: Tuple[List, List]=None) -> torch.Tensor:
        src_indices = [src_embedding['gene_to_index'][geneid] for geneid in edge_index[0]
                       if geneid in src_embedding['gene_to_index']]
        dst_indices = [dst_embedding['gene_to_index'][geneid] for geneid in edge_index[1]
                       if geneid in dst_embedding['gene_to_index']]

        src_embeddings = src_embedding['embedding'][src_indices]
        dst_embeddings = dst_embedding['embedding'][dst_indices]

        if self.method=='cosine':
            src_norm = torch.sqrt(torch.sum(src_embeddings ** 2, dim=1, keepdim=True))
            dst_norm = torch.sqrt(torch.sum(dst_embeddings ** 2, dim=1, keepdim=True))

            src_norm = torch.clamp(src_norm, min=1e-8)
            dst_norm = torch.clamp(dst_norm, min=1e-8)

            src_vecs = src_embeddings / src_norm
            dst_vecs = dst_embeddings / dst_norm

            scores = torch.sum(src_vecs * dst_vecs, dim=1)
        elif self.method=='dot':
            scores = torch.sum(src_embeddings * dst_embeddings, dim=1)
            scores = torch.sigmoid(scores)

        return scores
