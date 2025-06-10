import logging
from utils import *
import torch
import pandas as pd
import os
def train_reg_epoch(model,
                    data_loader,
                    optimizer,
                    device,
                    epoch,
                    num_epochs,
                    edge_loss_rate=0.5,
                    save_dir="model_outputs",
                    idx_limit=500,
                    epoch_limit=1,
                    epoch_cut=1,
                    cor_limit=0.2,
                    auc_limit=0.8,
                    auc_TF_CRE_limit=0.7,
                    auc_CRE_CRE_limit=0.7,
                    aupr_limit=0.7,
                    tf_metrics_interval=10,
                    tf_evaluation_data=None,
                    edge_type_weights=None,
                    neg_epoch=None,
                    mean_aupr_limit=None,
                    neg_epoch_limit=None,
                    node_label_dict=None):
    """
    单次训练函数
    """
    mes_criterion = torch.nn.MSELoss()
    bce_criterion = torch.nn.BCELoss()
    model.train()
    base_data = data_loader.dataset.base_data.clone()

    os.makedirs(save_dir, exist_ok=True)

    correlation_dict={}
    embedding_dict={}

    for idx, (train_neg, test_neg) in enumerate(data_loader):
    # train_neg, test_neg = data_loader.dataset.neg_samples[0]
    # train_neg = {k: v.to(device) for k, v in train_neg.items()}
    # test_neg = {k: v.to(device) for k, v in test_neg.items()}
    # idx=1

        train_neg = {k: v.squeeze(0).to(device) for k, v in train_neg.items()}
        test_neg = {k: v.squeeze(0).to(device) for k, v in test_neg.items()}

        base_data.train_neg_edge_dict = train_neg
        base_data.test_neg_edge_dict = test_neg
        base_data = base_data.to(device)

        optimizer.zero_grad()

        # Forward pass and training
        outputs = model(base_data)

        train_node_loss = 0
        train_cor_loss = 0

        batch_train_metrics = {}

        # Process node regression training predictions
        for node_type, pred in outputs['expression_dict'].items():
            train_mask = base_data.regression_train_mask_dict[node_type]
            if train_mask.any():
                true_values = base_data.x_dict[node_type][train_mask].clone()
                pred_values = pred[train_mask]

                # Calculate Pearson correlation loss
                train_cor_loss_, _ = pearson_loss(pred_values, true_values)
                train_cor_loss+=train_cor_loss_

                train_node_loss_ = mes_criterion(
                    true_values,
                    pred_values
                )
                train_node_loss += train_node_loss_

                batch_train_metrics[node_type] = {
                    'loss': train_node_loss_.item(),
                    'correlation': 1 - train_cor_loss_.item()
                }

        # Process edge classification training predictions
        train_edge_loss, train_preds, train_labels = compute_edge_metrics(
            model,
            outputs,
            base_data.train_edge_index_dict,
            base_data.train_neg_edge_dict,
            device,
            bce_criterion,
            edge_type_weights=edge_type_weights
        )


        train_loss = train_node_loss+edge_loss_rate*train_edge_loss
        # train_loss = train_edge_loss
        # train_loss = train_cor_loss

        # Backward pass
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Test evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(base_data)
            batch_test_metrics = {}
            test_node_loss = 0

            # Process node regression testing predictions
            for node_type, pred in test_outputs['expression_dict'].items():
                test_mask = base_data.regression_test_mask_dict[node_type]
                if test_mask.any():
                    true_values = base_data.x_dict[node_type][test_mask].clone()

                    pred_values = pred[test_mask]

                    # Calculate test loss
                    test_cor_loss_, correlations = pearson_loss(pred_values, true_values)

                    test_node_loss_ = mes_criterion(
                        true_values,
                        pred_values
                    )
                    test_node_loss+=test_node_loss_

                    batch_test_metrics[node_type] = {
                        'loss': test_node_loss_.item(),
                        'mean_correlation': 1 - test_cor_loss_.item(),
                        'correlation':correlations
                    }

            # Process edge classification testing predictions
            test_edge_loss, test_preds, test_labels, test_auc_dict, test_aupr_dict = compute_edge_metrics(
                model,
                test_outputs,
                base_data.test_edge_index_dict,
                base_data.test_neg_edge_dict,
                device,
                bce_criterion,
                edge_type_weights=edge_type_weights)

            # 随机抽取先验网络里的边，看看预测的边是否在先验边中有偏好
            # for edge_type in base_data.prior_edge_index_dict.keys():
            #     if str(edge_type) in test_neg and edge_type in base_data.prior_edge_attr_dict:
            #         src_type, _, dst_type = edge_type
            #
            #         # Get number of negative edges for this type
            #         num_neg_edges = test_neg[str(edge_type)].size(1)
            #
            #         # Get positive edges from prior edges where attr[:, -1] == 1
            #         prior_edge_index = base_data.prior_edge_index_dict[edge_type]
            #         prior_edge_attr = base_data.prior_edge_attr_dict[edge_type]
            #         pos_mask = prior_edge_attr[:, -2] != 1
            #         pos_edges = prior_edge_index[:, pos_mask]
            #         # pos_edges = prior_edge_index
            #
            #         # Randomly sample same number of positive edges
            #         if pos_edges.size(1) > num_neg_edges:
            #             perm = torch.randperm(pos_edges.size(1))
            #             sampled_pos_edges = pos_edges[:, perm[:num_neg_edges]]
            #         else:
            #             sampled_pos_edges = pos_edges
            #
            #         # Perform sampling based on edge type, similar to sample_negative_edges
            #         if src_type == 'CRE' and dst_type == 'CRE':
            #             # 对CRE-CRE边按染色体进行负采样
            #             if node_label_dict and src_type in node_label_dict:
            #                 cre_labels = node_label_dict[src_type]
            #
            #                 # 提取染色体信息
            #                 chromosomes = np.array([label.split('-')[0] for label in cre_labels.values()])
            #                 unique_chromosomes = np.unique(chromosomes)
            #
            #                 all_neg_edges = []
            #
            #                 # 对每个染色体单独处理
            #                 for chrom in unique_chromosomes:
            #                     chrom_indices = np.where(chromosomes == chrom)[0]
            #                     if len(chrom_indices) > 1:
            #                         # 创建局部到全局索引的映射
            #                         idx_to_local = {idx: local_idx for local_idx, idx in enumerate(chrom_indices)}
            #                         local_to_idx = {local_idx: idx for idx, local_idx in idx_to_local.items()}
            #
            #                         # 找出当前染色体内的正边
            #                         sampled_pos_np = sampled_pos_edges.cpu().numpy()
            #                         src_matches = np.isin(sampled_pos_np[0], chrom_indices)
            #                         dst_matches = np.isin(sampled_pos_np[1], chrom_indices)
            #                         mask = src_matches & dst_matches
            #
            #                         if mask.any():
            #                             chrom_pos_edges = sampled_pos_edges[:, torch.from_numpy(mask)]
            #
            #                             if chrom_pos_edges.size(1) > 0:
            #                                 # 转换为局部索引
            #                                 local_pos_edges = torch.tensor([
            #                                     [idx_to_local[idx.item()] for idx in chrom_pos_edges[0]],
            #                                     [idx_to_local[idx.item()] for idx in chrom_pos_edges[1]]
            #                                 ])
            #
            #                                 # 计算负样本数量
            #                                 local_neg_sample_size = chrom_pos_edges.size(1)
            #
            #                                 if local_neg_sample_size > 0:
            #                                     # 在染色体内生成负样本
            #                                     local_neg_edges = negative_sampling(
            #                                         edge_index=local_pos_edges,
            #                                         num_nodes=(len(chrom_indices), len(chrom_indices)),
            #                                         num_neg_samples=local_neg_sample_size
            #                                     )
            #
            #                                     # 转换回全局索引
            #                                     global_neg_edges = torch.tensor([
            #                                         [local_to_idx[idx.item()] for idx in local_neg_edges[0]],
            #                                         [local_to_idx[idx.item()] for idx in local_neg_edges[1]]
            #                                     ])
            #
            #                                     all_neg_edges.append(global_neg_edges)
            #
            #                 # 添加少量全局负样本（20%）
            #                 global_neg_sample_size = int(sampled_pos_edges.size(1) * 0.2)
            #                 if global_neg_sample_size > 0:
            #                     global_neg_edges = negative_sampling(
            #                         edge_index=sampled_pos_edges,
            #                         num_nodes=(base_data.x_dict[src_type].size(0),
            #                                    base_data.x_dict[dst_type].size(0)),
            #                         num_neg_samples=global_neg_sample_size
            #                     )
            #                     all_neg_edges.append(global_neg_edges)
            #
            #                 if all_neg_edges:
            #                     sampled_neg_edges = torch.cat(all_neg_edges, dim=1)
            #                 else:
            #                     # 如果染色体负采样失败，则使用常规负采样
            #                     sampled_neg_edges = negative_sampling(
            #                         edge_index=sampled_pos_edges,
            #                         num_nodes=(base_data.x_dict[src_type].size(0),
            #                                    base_data.x_dict[dst_type].size(0)),
            #                         num_neg_samples=num_neg_edges
            #                     )
            #             else:
            #                 # 如果没有染色体信息，使用常规负采样
            #                 sampled_neg_edges = negative_sampling(
            #                     edge_index=sampled_pos_edges,
            #                     num_nodes=(base_data.x_dict[src_type].size(0),
            #                                base_data.x_dict[dst_type].size(0)),
            #                     num_neg_samples=num_neg_edges
            #                 )
            #
            #         elif src_type == 'TF' and dst_type == 'CRE':
            #             # 对TF-CRE边使用TF感知的负采样策略
            #             unique_TFs = torch.unique(sampled_pos_edges[0]).cpu().numpy()
            #
            #             if len(unique_TFs) > 0:
            #                 idx_to_local = {idx: local_idx for local_idx, idx in enumerate(unique_TFs)}
            #                 local_to_idx = {local_idx: idx for idx, local_idx in idx_to_local.items()}
            #
            #                 local_pos_edges = torch.tensor([
            #                     [idx_to_local[idx.item()] for idx in sampled_pos_edges[0]],
            #                     sampled_pos_edges[1].tolist()
            #                 ])
            #
            #                 sampled_neg_edges = negative_sampling(
            #                     edge_index=local_pos_edges,
            #                     num_nodes=(len(unique_TFs), base_data.x_dict[dst_type].size(0)),
            #                     num_neg_samples=sampled_pos_edges.size(1)
            #                 )
            #
            #                 sampled_neg_edges = torch.tensor([
            #                     [local_to_idx[local_idx.item()] for local_idx in sampled_neg_edges[0]],
            #                     sampled_neg_edges[1].tolist()
            #                 ])
            #             else:
            #                 # 如果没有TFs，使用常规负采样
            #                 sampled_neg_edges = negative_sampling(
            #                     edge_index=sampled_pos_edges,
            #                     num_nodes=(base_data.x_dict[src_type].size(0),
            #                                base_data.x_dict[dst_type].size(0)),
            #                     num_neg_samples=sampled_pos_edges.size(1)
            #                 )
            #         else:
            #             # 对其他边类型使用标准负采样
            #             sampled_neg_edges = negative_sampling(
            #                 edge_index=sampled_pos_edges,
            #                 num_nodes=(base_data.x_dict[src_type].size(0),
            #                            base_data.x_dict[dst_type].size(0)),
            #                 num_neg_samples=sampled_pos_edges.size(1)
            #             )
            #
            #         # Get predictions for both positive and negative edges
            #         edge_dict = {edge_type: torch.cat([sampled_pos_edges, sampled_neg_edges], dim=1)}
            #         preds = model.edge_decoder(outputs['z_dict'], edge_dict)
            #
            #         # Create labels (1 for positive edges, 0 for negative edges)
            #         labels = torch.cat([
            #             torch.ones(sampled_pos_edges.size(1)),
            #             torch.zeros(sampled_neg_edges.size(1))
            #         ]).to(device)
            #
            #         print(f"ramdom auc {edge_type}  {calculate_auc(preds[str(edge_type)].cpu(), labels.cpu())}")

            if tf_evaluation_data and tf_metrics_interval > 0 and (idx + 1) % tf_metrics_interval == 0:

                current_tf_metrics = evaluate_tf_cre_metrics(
                    model=model,
                    data=base_data,
                    tf_evaluation_data=tf_evaluation_data,
                    device=device
                )

                if "average" in current_tf_metrics:
                    avg = current_tf_metrics["average"]
                    logging.info(
                        f"Epoch {epoch + 1} - TF Metrics - Average AUPR: {avg['aupr']:.4f}, Average F1: {avg['f1']:.4f}, TFs evaluated: {avg['num_tfs']}")

        model.train()

        train_auc = calculate_auc(train_preds.cpu(), train_labels.cpu())
        train_aupr = calculate_aupr(train_preds.cpu(), train_labels.cpu())
        test_auc = calculate_auc(test_preds.cpu(), test_labels.cpu())
        test_aupr = calculate_aupr(test_preds.cpu(), test_labels.cpu())

        # 打印测试结果
        log_str = [f"Epoch {epoch + 1}/{num_epochs} Batch {idx + 1}/{len(data_loader)} neg_epoch {neg_epoch}"]

        log_str.append("Training:")
        for node_type, metrics in batch_train_metrics.items():
            log_str.append(f"{node_type} Loss: {metrics['loss']:.4f}")
            log_str.append(f"{node_type} Correlation: {metrics['correlation']:.4f}")

        log_str.append(f" Loss: {train_edge_loss.item():.4f}"
                       f" auc: {train_auc:.4f}"
                       f" aupr: {train_aupr:.4f}")

        log_str.append("Testing:")
        for node_type, metrics in batch_test_metrics.items():
            log_str.append(f"{node_type} Loss: {metrics['loss']:.4f}")
            log_str.append(f"{node_type} Correlation: {metrics['mean_correlation']:.4f}")

        log_str.append(f" Loss: {test_edge_loss.item():.4f}"
                       f" auc: {test_auc:.4f}"
                       f" aupr: {test_aupr:.4f}")

        logging.info(" | ".join(log_str))

        # 判断是否满足早停条件
        first_correlation = batch_test_metrics[list(batch_test_metrics.keys())[0]]['mean_correlation']

        # if ((idx > idx_limit or epoch > epoch_limit) and
        #     first_correlation > cor_limit and
        #     test_auc > auc_limit and
        #     test_aupr > aupr_limit) or \
        #     ((idx + 1) % tf_metrics_interval == 0 and
        #     avg['aupr']>mean_aupr_limit and
        #         neg_epoch>=neg_epoch_limit) or \
        #         epoch>epoch_cut:
        if  ((idx + 1) % tf_metrics_interval) == 0 and \
            avg['aupr']>mean_aupr_limit and \
                neg_epoch>=neg_epoch_limit and \
                test_auc_dict["('TF', 'edge', 'CRE')"]>auc_TF_CRE_limit and \
                    first_correlation > cor_limit:
            # and test_auc_dict["('CRE', 'edge', 'CRE')"]>auc_CRE_CRE_limit
            for node_type in base_data.x_dict.keys():
                if node_type in batch_test_metrics:
                    metrics = batch_test_metrics[node_type]
                    test_mask = base_data.regression_test_mask_dict[node_type]
                    node_indices = torch.where(test_mask)[0].cpu().numpy()
                    correlation_df = pd.DataFrame({
                        'node_id': node_indices,
                        'correlation': metrics['correlation'].cpu().numpy()
                    })
                    correlation_dict[node_type] = correlation_df

                embeddings = test_outputs['z_dict'][node_type].cpu().numpy()
                embedding_df = pd.DataFrame(
                    embeddings,
                    columns=[f'dim_{i}' for i in range(embeddings.shape[1])]
                )
                embedding_df['node_id'] = range(len(embedding_df))
                embedding_dict[node_type] = embedding_df

            torch.save(model.state_dict(), os.path.join(save_dir, 'model_parameters.pt'))
            return correlation_dict, embedding_dict
    return correlation_dict, embedding_dict

def train_reg_model(model,
                    optimizer,
                    data_loader,
                    device,
                    num_epochs,
                    edge_loss_rate,
                    idx_limit=500,
                    epoch_limit=1,
                    epoch_cut=1,
                    cor_limit=0.2,
                    auc_limit=0.8,
                    aupr_limit=0.7,
                    auc_TF_CRE_limit=0.7,
                    auc_CRE_CRE_limit=0.7,
                    tf_metrics_interval=10,
                    tf_evaluation_data=None,
                    edge_type_weights=None,
                    mean_aupr_limit=None,
                    neg_epoch_limit=None,
                    neg_epoch=None,
                    node_label_dict=None):
    """
    总训练函数
    """

    for epoch in range(num_epochs):
        correlation_dict, embedding_dict = train_reg_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            edge_loss_rate=edge_loss_rate,
            idx_limit=idx_limit,
            epoch_limit=epoch_limit,
            epoch_cut=epoch_cut,
            cor_limit=cor_limit,
            auc_limit=auc_limit,
            aupr_limit=aupr_limit,
            auc_TF_CRE_limit=auc_TF_CRE_limit,
            auc_CRE_CRE_limit=auc_CRE_CRE_limit,
            tf_metrics_interval=tf_metrics_interval,
            tf_evaluation_data=tf_evaluation_data,
            edge_type_weights=edge_type_weights,
            mean_aupr_limit=mean_aupr_limit,
            neg_epoch_limit=neg_epoch_limit,
            neg_epoch=neg_epoch,
            node_label_dict=node_label_dict
        )
        if correlation_dict:
            return correlation_dict, embedding_dict
    return None, None
