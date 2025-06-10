import os
from utils import *

config_path = "config.yaml"
config = load_config(config_path)

from driver_regulators.driver_regulators import *
name = "K562"
atten_node_df = pd.read_csv('~/workspace/algorithms_raw/data/atten_node_df_K562.csv')
grn_df = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/hummus/multiplex/CRE/CRE.csv',
                                        sep='\t',
                                        header=None,
                                        names=['from', 'to', 'score'])
clusters = atten_node_df['cluster'].unique()

all_results = []

# 对每个子群进行分析
for cluster_id in clusters:
    cluster_genes = atten_node_df[atten_node_df['cluster'] == cluster_id]['name'].tolist()

    cluster_edges = grn_df[
        grn_df['from'].isin(cluster_genes) &
        grn_df['to'].isin(cluster_genes)
        ]

    G = nx.DiGraph()
    G.add_nodes_from(cluster_genes)

    for _, row in cluster_edges.iterrows():
        G.add_edge(row['from'], row['to'], weight=row['score'])

    print(f"  网络节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    if G.number_of_edges() == 0:
        continue

    cluster_atten_df = atten_node_df[atten_node_df['cluster'] == cluster_id].set_index('name')
    influence_scores = pd.DataFrame(index=cluster_genes)
    influence_scores['influence_score'] = cluster_atten_df.loc[cluster_genes, 'atten_degree']
    influence_scores['score_out'] = influence_scores['influence_score']
    influence_scores['score_in'] = influence_scores['influence_score']

    influence_scores.fillna(0, inplace=True)

    root_nodes_set = root_nodes(G)
    end_nodes_set = end_nodes(G)

    # MDS
    try:
        mds_driver_set, mds_intermittent_nodes = MDScontrol(G.copy(), solver='SCIP')
    except Exception as e:
        mds_driver_set = set()
        mds_intermittent_nodes = set()

    # MFV
    try:
        mfvs_driver_set, source_nodes = MFVScontrol(G.copy(), influence_scores.loc[:, 'influence_score'], solver='SCIP')
    except Exception as e:
        mfvs_driver_set = set()
        source_nodes = set()

    all_drivers = mds_driver_set.union(mfvs_driver_set)
    common_drivers = mds_driver_set.intersection(mfvs_driver_set)

    for gene in all_drivers:
        if gene in influence_scores.index:
            influence_score = influence_scores.loc[gene, 'influence_score']
        else:
            influence_score = None

        all_results.append({
            'cluster': cluster_id,
            'gene': gene,
            'influence_score': influence_score,
            'is_mds_driver': gene in mds_driver_set,
            'is_mfvs_driver': gene in mfvs_driver_set,
            'is_common_driver': gene in common_drivers,
            'is_root_node': gene in root_nodes_set,
            'is_end_node': gene in end_nodes_set
        })

# 创建所有结果的DataFrame
all_drivers_df = pd.DataFrame(all_results)

# 保存所有驱动基因信息
all_drivers_df.to_csv(os.path.join(config['dataset']['processed_dir'], 'all_cluster_drivers_{}.csv'.format(name)), index=False)

