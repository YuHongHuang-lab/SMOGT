# MM (Melanoma)
output_dir = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma'
models_and_networks_dir = os.path.join(output_dir, "models_and_networks_p")
os.makedirs(models_and_networks_dir, exist_ok=True)
all_nodes_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma/raw_600/graph_0_nodes.csv'
train_nodes_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma/train_id.csv'
node_names_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma/raw_600/graph_0_node_names.csv'
edges_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma/GRN_df_p.csv'
gene_names_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma/predicted_genes.txt'

common_seacells_mat_train = pd.read_csv(
    train_nodes_path,
    index_col=0)
train_id = common_seacells_mat_train['x'].tolist()
del common_seacells_mat_train

common_seacells_mat = pd.read_csv(all_nodes_path,
                                  index_col=0)

# Generate training file
train_df = pd.DataFrame({'id' : common_seacells_mat.columns,
                        'train' : [id in train_id for id in common_seacells_mat.columns]})
train_df['train'] = train_df['train'].astype(int)

Genes_Peaks_df = pd.read_csv(node_names_path,
                             index_col=0)
TF = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='TF', 'name'].to_list()
Target = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='Target', 'name'].to_list()
CRE = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='CRE', 'name'].to_list()

GRN_df = pd.read_csv(edges_path)

# Prepare expression matrices
common_seacells_mat_TF = common_seacells_mat.loc[TF, ]
common_seacells_mat_Target = common_seacells_mat.loc[Target, ]
common_seacells_mat_CRE = common_seacells_mat.loc[CRE, ]

# Prepare network data
TF_TF_network = GRN_df.loc[GRN_df['edge_id_type']=='TF_TF', ['from', 'to']]
TF_TF_network['score'] = 1

CRE_Target_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_Target', ['from', 'to']]
CRE_Target_network['score'] = 1

CRE_TF_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_TF', ['from', 'to']]
CRE_TF_network['score'] = 1

TF_CRE_network = GRN_df.loc[GRN_df['edge_id_type']=='TF_CRE', ['from', 'to']]
TF_CRE_network['score'] = 1

CRE_CRE_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_CRE', ['from', 'to']]
CRE_CRE_network['score'] = 1


global_network = GlobalNetworkManager(
    TF_TF_network=TF_TF_network,
    CRE_CRE_network=CRE_CRE_network,
    TF_CRE_network=TF_CRE_network,
    CRE_TF_network=CRE_TF_network,
    CRE_Target_network=CRE_Target_network,
    TF_names=TF,
    CRE_names=CRE,
    Target_names=Target
)

predictor = GeneExpressionPredictor(global_network)

predictor.prepare_data(tf_expr=common_seacells_mat_TF,
                       cre_expr=common_seacells_mat_CRE,
                       target_expr=common_seacells_mat_Target,
                       train_df=train_df)

gene_names = pd.read_csv(gene_names_path,
                         header=None, names=['gene'])
gene_names = gene_names['gene'].tolist()


gene_prediction = pd.read_csv(os.path.join(output_dir, 'gene_prediction_our_parallel.csv'))
gene_list = gene_prediction.loc[gene_prediction['correlation']>0, 'gene']
gene_list = gene_list.tolist()

adata_RNA_MM_seacells_600_ad = sc.read_h5ad('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/seob_RNA_MM_seacells_600_ad_reduce.h5ad')

gene_predict_df = simulate_perturbation(perturbations={'chr16-85882146-85883060':2},
                                        predictor=predictor,
                                        original_tf_expr_df=common_seacells_mat_TF,
                                        original_cre_expr_df=common_seacells_mat_CRE,
                                        models_dir=models_and_networks_dir,
                                        gene_list=gene_list,
                                        ncores=1)

# Perform multiple iterations to simulate long-range regulatory effects of the network
common_seacells_mat_TF_new = common_seacells_mat_TF.copy()
common_seacells_mat_TF_new.update(gene_predict_df)

gene_predict_df = simulate_perturbation(perturbations={'chr16-85882146-85883060':2},
                                        predictor=predictor,
                                        original_tf_expr_df=common_seacells_mat_TF_new,
                                        original_cre_expr_df=common_seacells_mat_CRE,
                                        models_dir=models_and_networks_dir,
                                        gene_list=gene_list,
                                        ncores=1)

gene_predict_df.to_csv(os.path.join(output_dir,'chr16-85882146-85883060_2.csv'))

# gene_predict_df = pd.read_csv(os.path.join(output_dir, 'gene_preturb_our_parallel_KLF1.csv'))


run_perturbation_analysis(
    perturbations={'KLF1':0},
    predictor=predictor,
    original_tf_expr_df=common_seacells_mat_TF,
    original_cre_expr_df=common_seacells_mat_CRE,
    original_target_expr_df=common_seacells_mat_Target,
    models_dir=models_and_networks_dir,
    adata=adata_RNA_MM_seacells_600_ad,
    output_dir=output_dir,
    ko_results_path=os.path.join(output_dir, 'chr16-85882146-85883060_2.csv'),
    gene_list=gene_names,
    embedding_key='X_umap',
    metadata_key='seurat_clusters',
    n_cpu=1,
    tf_name='chr16-85882146-85883060_2')


# Analyze KLF1 network weights
learned_network_df = predictor.analyze_weights(
    gene_name='KLF1',
    model_path=os.path.join(output_dir, 'models_and_networks_p/KLF1_model.pth')
)

learned_network_df.to_csv(os.path.join(output_dir, 'GRN_KLF1.csv'))


# Analyze IRF8 network weights
learned_network_df = predictor.analyze_weights(
    gene_name='IRF8',
    model_path=os.path.join(output_dir, 'models_and_networks_p/IRF8_model.pth')
)

learned_network_df.to_csv(os.path.join(output_dir, 'GRN_IRF8.csv'))
