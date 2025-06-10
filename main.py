import gc
from copy import deepcopy

import torch.cuda
import torch.optim as optim
from dataset.bio_dataset import *
from model.GraphAutoencoder import *
from train import *
from utils import *

if __name__ == '__main__':
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

    dataset = BioDataset(config=config["dataset"])
    data = dataset.data
    data = split_edges(
        data,
        rate=config['dataset'].get('rate', 0.2),
        seed=config['dataset'].get('seed', 42),
        mask_prior=config['dataset'].get('mask_prior', False),
        prior_mask_rate=0,
        use_last_col=False
    )

    metadata = dataset.get_metadata()

    k_fold_splits = split_nodes(
        data,
        rate=config['dataset'].get('regression_rate', 0.2),
        seed=config['dataset'].get('seed', 42),
        k_fold=config['dataset']['k_fold']
    )

    tf_metrics_interval = config.get('tf_metrics', {}).get('interval', 10)
    tf_min_positive_edges = config.get('tf_metrics', {}).get('min_positive_edges', 100)

    tf_evaluation_data = prepare_tf_cre_evaluation_data(
        data=data,
        min_positive_edges=tf_min_positive_edges
    )

    edge_type_weights = config.get('model', {}).get('edge_type_weights', {})

    neg_sample_paths = create_data_with_neg_samples_in_batches(config=config)
    neg_sample_paths = neg_sample_paths[:5]

    mean_aupr_limit = config['training']['mean_aupr_limit']
    neg_epoch_limit = config['training']['neg_epoch_limit']

    node_label_dict = {}
    for node_type in data.node_types:
        node_label_dict[node_type] = {
            i: label for i, label in enumerate(data[node_type].label)
        }

    fold_results = None
    accumulate_results = None
    for fold in range(config['dataset']['k_fold']):
        fold_splits = {node_type: mask[fold] for node_type, mask in k_fold_splits.items()}
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
        for neg_epoch, neg_sample_path in enumerate(neg_sample_paths):
            data_loader = create_data_loader(config=config,
                                             device=device,
                                             data=data,
                                             fold_splits=fold_splits,
                                             neg_sample_path=neg_sample_path)

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

                # 添加基因id
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