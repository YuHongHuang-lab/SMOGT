# import torch.optim as optim
# from dataset.bio_dataset import *
# from model.GraphAutoencoder import *
# from train import *
# from utils import *
#
# config_path="config.yaml"
# config = load_config(config_path)
# logging.basicConfig(
#         level=logging.INFO if config["log"]["level"] == "INFO" else logging.DEBUG,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
# logging.info(config)
#
# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")
# if torch.cuda.is_available():
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("Using CPU")
#
# # 创建并处理数据集
# dataset = BioDataset(config=config["dataset"])
# data = dataset.data
# data = split_edges(
#     data,
#     rate=config['dataset'].get('rate', 0.2),
#     seed=config['dataset'].get('seed', 42),
#     mask_prior=config['dataset'].get('mask_prior', False),
#     prior_mask_rate=0,
#     use_last_col=True
# )
#
# metadata = dataset.get_metadata()
#
# k_fold_splits = split_nodes(
#     data,
#     rate=config['dataset'].get('regression_rate', 0.2),
#     seed=config['dataset'].get('seed', 42),
#     k_fold=config['dataset']['k_fold']
# )
#
# fold_results = []
# for fold in range(config['dataset']['k_fold']):
#     fold_splits = {node_type:mask[fold] for node_type, mask in k_fold_splits.items()}
#     data_loader = create_data_loader(config=config,
#                                      device=device,
#                                      data=data,
#                                      fold_splits=fold_splits)
#
#     # Initialize model
#     # model = SimplifiedHGTEncoder(
#     #     metadata=metadata,
#     #     hidden_dim=config['model']['hidden_dim'],
#     #     embedding_dim=config['model']['embedding_dim'],
#     #     heads=config['attention'].get('heads', 1)
#     # )
#
#     model = SimplifiedGraphAtten_regression(
#         metadata=metadata,
#         hidden_dim=config['model']['hidden_dim'],
#         embedding_dim=config['model']['embedding_dim'],
#         heads=config['attention'].get('heads', 1),
#         dropout=config['model']['dropout'],
#         raw_dim=data.x_dict['TF'].size(1),
#         layer_nums=config['model']['num_layers']
#     )
#
#
#     # model = SimplifiedGraphAutoencoder(
#     #     metadata=metadata,
#     #     hidden_dim=config['model']['hidden_dim'],
#     #     embedding_dim=config['model']['embedding_dim'])
#
#     model = model.to(device)
#
#     # Initialize optimizer
#     optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
#
#     # Train model
#     correlation_dict_, embedding_dict_ = train_reg_model(
#         model=model,  # 传入基础数据
#         optimizer=optimizer,
#         data_loader=data_loader,
#         device=device,
#         num_epochs=config["training"]["epochs"],
#         edge_loss_rate=config["model"]["edge_loss_rate"],
#         idx_limit=config["training"]["idx_limit"],
#         epoch_limit=config["training"]["epoch_limit"],
#         cor_limit=config["training"]["cor_limit"],
#         auc_limit=config["training"]["auc_limit"],
#         aupr_limit=config["training"]["aupr_limit"]
#     )
#
#     if correlation_dict_:
#         fold_results.append((correlation_dict_, embedding_dict_))
#
# if fold_results:
#     final_correlation_dict, final_embedding_dict, avg_embedding_dict = merge_fold_results(
#         fold_results
#     )
#
#     for node_type in final_correlation_dict.keys():
#         labels = data[node_type].label
#
#         # Update correlation dict
#         cor_df = final_correlation_dict[node_type]
#         cor_df['gene_id'] = cor_df['node_id'].apply(lambda x: labels[x])
#         cor_df = cor_df.drop('node_id', axis=1)
#         final_correlation_dict[node_type] = cor_df
#
#         # Update embedding dict - both final and avg will be updated and merged
#         avg_df = avg_embedding_dict[node_type]
#         avg_df['gene_id'] = avg_df['node_id'].apply(lambda x: labels[x])
#         avg_df = avg_df.drop('node_id', axis=1)
#         avg_embedding_dict[node_type] = avg_df
#
#     cor_df = pd.concat([final_correlation_dict['TF'], final_correlation_dict['Target']], axis=0)
#     avg_df = pd.concat([avg_embedding_dict['TF'], avg_embedding_dict['Target']], axis=0)
#
#     cor_df.to_csv(os.path.join(config['dataset']['processed_dir'], f"cor_df_{config['dataset']['seed']}.csv"))
#     avg_df.to_csv(os.path.join(config['dataset']['processed_dir'], f"avg_df_{config['dataset']['seed']}.csv"))
#
#
#
