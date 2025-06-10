#A549 predicted
config_path = "config.yaml"
config = load_config(config_path)

EBuider = EvaluationBuider(config=config,
                           embedding_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_A549_predicted.csv',
                           negative_ratio=1)
EBuider.load_embedding()
result = EBuider.analysis(edge_types=['TF-CRE', 'CRE-CRE'])

#buide network
buider = NetworkBuider(config=config,
                       edge_types=[['TF','TF'], ['TF','CRE'], ['CRE', 'CRE'],
                                   ['CRE', 'Target'], ['CRE', 'TF'], ['Target', 'Target']],
                       threshold={'TF_TF':0.999,
                         'TF_CRE':0.8,
                         'CRE_CRE':0.9,
                         'CRE_Target':0.2,
                         'CRE_TF':0.2,
                         'Target_Target':0.999},
                       method='dot',
                       topp_CRE=0.01)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_A549_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/A549_predict/processed/hummus_A549')
