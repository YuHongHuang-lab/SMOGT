import gc
from copy import deepcopy

import torch.cuda
import torch.optim as optim
from dataset.bio_dataset import *
from model.GraphAutoencoder import *
from train import *
from utils import *

if __name__ == '__main__':
    # Load configuration file
    config_path = "config.yaml"
    config = load_config(config_path)
    logging.basicConfig(
        level=logging.INFO if config["log"]["level"] == "INFO" else logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Load dataset
    dataset = BioDataset(config=config["dataset"])
    data = dataset.data

    # Split the edges into training and test sets
    data = split_edges(
        data,
        rate=config['dataset'].get('rate', 0.2),
        seed=config['dataset'].get('seed', 42),
        mask_prior=config['dataset'].get('mask_prior', False),
        prior_mask_rate=0,
        use_last_col=False
    )

    # Load node information
    metadata = dataset.get_metadata()

    # Split TF and TG into training and test sets
    k_fold_splits = split_nodes(
        data,
        rate=config['dataset'].get('regression_rate', 0.2),
        seed=config['dataset'].get('seed', 42),
        k_fold=config['dataset']['k_fold']
    )

    # Load the correct TF-CRE relationships from the prior network,
    # i.e., the last line in the edge annotation data — edge_id_T,
    # to print AUPR of TF-CRE in real time during training
    tf_metrics_interval = config.get('tf_metrics', {}).get('interval', 10)
    tf_min_positive_edges = config.get('tf_metrics', {}).get('min_positive_edges', 100)

    tf_evaluation_data = prepare_tf_cre_evaluation_data(
        data=data,
        min_positive_edges=tf_min_positive_edges
    )

    # Load the proportion of different edge types in the loss function during edge prediction
    edge_type_weights = config.get('model', {}).get('edge_type_weights', {})

    # Load the negative edge sampling datase
    neg_sample_paths = create_data_with_neg_samples_in_batches(config=config)
    neg_sample_paths = neg_sample_paths[:5]

    # Load early stopping conditions
    mean_aupr_limit = config['training']['mean_aupr_limit']
    neg_epoch_limit = config['training']['neg_epoch_limit']

    # Load node types
    node_label_dict = {}
    for node_type in data.node_types:
        node_label_dict[node_type] = {
            i: label for i, label in enumerate(data[node_type].label)
        }

    # Start training
    fold_results = None
    accumulate_results = None
    # k_fold
    for fold in range(config['dataset']['k_fold']):
        fold_splits = {node_type: mask[fold] for node_type, mask in k_fold_splits.items()}
        # Model initialization
        model = SimplifiedGraphAtten_regression(
            metadata=metadata,
            hidden_dim=config['model']['hidden_dim'],
            embedding_dim=config['model']['embedding_dim'],
            heads=config['attention'].get('heads', 1),
            dropout=config['model']['dropout'],
            raw_dim=data.x_dict['TF'].size(1),
            layer_nums=config['model']['num_layers'],
            top_k_type=config['attention']['top_k_type'],
            top_k=config['attention']['top_k']
        )

        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

        correlation_dict_ = None
        embedding_dict_ = None
        # To reduce the memory burden, the negative sample set is loaded in batches
        for neg_epoch, neg_sample_path in enumerate(neg_sample_paths):
            data_loader = create_data_loader(config=config,
                                             device=device,
                                             data=data,
                                             fold_splits=fold_splits,
                                             neg_sample_path=neg_sample_path)
            # Check the early stopping condition in each epoch.
            # If the condition is met, neither correlation_dict_ nor embedding_dict_ will be empty
            # During the actual training process, the K-fold mode is not strictly followed.
            # In each fold of training, results are generated once the early stopping condition is reached
            # Researchers can export the low-dimensional representation matrix of nodes based on the actual situation (saved in processed_dir with the name avg_df_{config['dataset']['seed']}.csv)
            # If training continues until the completion of K folds,
            # the low-dimensional node representation is the average after K folds
            correlation_dict_, embedding_dict_ = train_reg_model(
                model=model,
                optimizer=optimizer,
                data_loader=data_loader,
                device=device,
                num_epochs=config["training"]["epochs"],
                edge_loss_rate=config["model"]["edge_loss_rate"],
                idx_limit=config["training"]["idx_limit"],
                epoch_limit=config["training"]["epoch_limit"],
                epoch_cut=config["training"]["epoch_cut"],
                cor_limit=config["training"]["cor_limit"],
                auc_limit=config["training"]["auc_limit"],
                aupr_limit=config["training"]["aupr_limit"],
                auc_TF_CRE_limit=config["training"]["auc_TF_CRE_limit"],
                auc_CRE_CRE_limit=config["training"]["auc_CRE_CRE_limit"],
                tf_metrics_interval=tf_metrics_interval,
                tf_evaluation_data=tf_evaluation_data,
                edge_type_weights=edge_type_weights,
                mean_aupr_limit=mean_aupr_limit,
                neg_epoch_limit=neg_epoch_limit,
                neg_epoch=neg_epoch,
                node_label_dict=node_label_dict
            )

            del data_loader
            torch.cuda.empty_cache()
            gc.collect()

            if correlation_dict_:
                fold_results = (correlation_dict_, embedding_dict_)
                accumulate_results = merge_fold_results(accumulate_results, fold_results, fold=fold)

                del correlation_dict_, embedding_dict_
                gc.collect()
                torch.cuda.empty_cache()

            if fold_results:
                correlation_dict, embedding_dict = deepcopy(accumulate_results)

                for node_type in data.node_types:
                    labels = data[node_type].label

                    # Restore the gene symbol ID
                    if node_type in correlation_dict:
                        cor_df = correlation_dict[node_type]
                        cor_df['gene_id'] = cor_df['node_id'].apply(lambda x: labels[x])
                        cor_df = cor_df.drop('node_id', axis=1)
                        correlation_dict[node_type] = cor_df

                    if node_type in embedding_dict:
                        emb_df = embedding_dict[node_type]
                        emb_df['gene_id'] = emb_df['node_id'].apply(lambda x: labels[x])
                        emb_df = emb_df.drop('node_id', axis=1)
                        embedding_dict[node_type] = emb_df

                cor_df = pd.concat([correlation_dict['TF'], correlation_dict['Target']], axis=0)
                avg_df = pd.concat([item for key, item in embedding_dict.items()], axis=0)

                cor_df.to_csv(
                    os.path.join(config['dataset']['processed_dir'], f"cor_df_{config['dataset']['seed']}.csv"))
                avg_df.to_csv(
                    os.path.join(config['dataset']['processed_dir'], f"avg_df_{config['dataset']['seed']}.csv"))

                del correlation_dict, embedding_dict, cor_df, avg_df
                gc.collect()
                torch.cuda.empty_cache()