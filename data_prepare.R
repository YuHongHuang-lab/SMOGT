# *load function----------------------------------------------------------------------
.libPaths('~/R/x86_64-pc-linux-gnu-library/4.3/')
# reticulate::use_condaenv("/mnt/data/home/tycloud/mambaforge/envs/dictys/")
# Sys.setenv(RETICULATE_PYTHON = "/mnt/data/home/tycloud/mambaforge/envs/dictys/bin/python")

reticulate::use_condaenv("/mnt/data/home/tycloud/software/anaconda3/envs/scFates/")
Sys.setenv(RETICULATE_PYTHON = "/mnt/data/home/tycloud/software/anaconda3/envs/scFates/bin/python3")

library(Signac)
library(Seurat)
# library(EnsDb.Hsapiens.v75)
library(EnsDb.Hsapiens.v86)
library(ggplot2)
library(patchwork)
library(tidyverse)
library(data.table)
library(slingshot)
library(homologene)
library(rtracklayer) 
library(igraph)

# source("~/workspace/algorithms_raw/data_prepare.R")
load('~/workspace/algorithms_raw_paper_4/data_prepare.RData')
source("~/workspace/scATAC/utils.R")
bad_genes = readRDS(file = "../scATAC/data/bad_genes.rds")
load(file = "~/database/Ligand_Receptor/Ligand_Receptor_Human.RData")
TF = homologs <- homologene(TF, inTax = 10090, outTax = 9606)
TF = TF$`9606`

TF_2 = read.table('~/database/allTFs_hg38.txt')
TF_2 = TF_2$V1

TF_3 = read.table(file = '~/database/Homo_sapiens_TF', 
                  header = T, 
                  sep = "\t", 
                  stringsAsFactors = F,
                  fill = T)
TF_3 = TF_3$Symbol
TF_3 = TF_3[TF_3!='']

TF = unique(c(TF,TF_2,TF_3))

ppi = readRDS(file = "~/database/ppi/BIOGRID_STRING_CEFCON_higher_600_Human.rds")

Protein_coding_Genes = genes(EnsDb.Hsapiens.v86,
                             filter = ~ gene_biotype == "protein_coding",
                             columns = c("gene_id", "gene_name", "gene_biotype"))
Protein_coding_Genes <- as.data.frame(Protein_coding_Genes)
Protein_coding_Genes = Protein_coding_Genes$gene_name


TF_Target = readRDS(file = "~/database/TF_Target/TF_Target_Human.rds")

TF_Target = rbind(ppi, TF_Target)
TF_Target <- unique(TF_Target, by = c("from", "to"))


counts <- Read10X_h5(filename = "~/workspace/algorithms_raw/data/PBMC/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5")
counts_atac = counts$Peaks

#**snRNA-----------------------------------------------------------------------------------
counts = counts$`Gene Expression`
seob_PBMC = CreateSeuratObject(counts = counts,
                               min.cells = 3,
                               min.features = 200)
rm(counts); gc()
seob_PBMC = RunSeurat(seob_PBMC, 
                      split_by=NULL,
                      coln_add='all',
                      cellid_add=T,
                      genes_add=NULL,
                      integrated="no",
                      integrated_assay="RNA",
                      cellid=NULL,
                      min_cells=3,
                      min_features=200,
                      max_features=0,
                      MT_filter = 20,
                      nfeatures=2000,
                      npcs=15,
                      resolution=0.6,
                      k_anchor=5,
                      k_filter=200,
                      k_score=30,
                      outfeatures=bad_genes,
                      min_batch_num=NULL,
                      k_weight=100)
seob_PBMC@meta.data$seurat_clusters = as.character(seob_PBMC@meta.data$seurat_clusters)
seob_PBMC@meta.data$celltype = as.character(seob_PBMC@meta.data$seurat_clusters)

DimPlot(seob_PBMC, reduction = "umap", group.by = "seurat_clusters",label = T)


#**snATAC-----------------------------------------------------------------------------------
chrom_assay <- CreateChromatinAssay(
  counts = counts_atac,
  sep = c(":", "-")
)
seob_ATAC_PBMC <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "peaks"
)
# granges(seob_ATAC_PBMC)
# extract gene annotations from EnsDb
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)

# change to UCSC style since the data was mapped to hg38
seqlevels(annotations) <- paste0('chr', seqlevels(annotations))
genome(annotations) <- "hg38"

Annotation(seob_ATAC_PBMC) <- annotations

comid = intersect(colnames(seob_ATAC_PBMC), colnames(seob_PBMC))

seob_ATAC_PBMC = subset(seob_ATAC_PBMC, cells=comid)
seob_PBMC = subset(seob_PBMC, cells=comid)


seob_ATAC_PBMC@meta.data$celltype = seob_PBMC@meta.data[rownames(seob_ATAC_PBMC@meta.data),
                                                        "celltype"]
seob_ATAC_PBMC <- RunTFIDF(seob_ATAC_PBMC)
seob_ATAC_PBMC <- FindTopFeatures(seob_ATAC_PBMC, min.cutoff = 'q75')
seob_ATAC_PBMC <- RunSVD(seob_ATAC_PBMC)

seob_ATAC_PBMC <- RunUMAP(object = seob_ATAC_PBMC, reduction = 'lsi', dims = 2:30)
seob_ATAC_PBMC <- FindNeighbors(object = seob_ATAC_PBMC, reduction = 'lsi', dims = 2:30)
seob_ATAC_PBMC <- FindClusters(object = seob_ATAC_PBMC, verbose = FALSE, algorithm = 3)

DimPlot(seob_ATAC_PBMC, reduction = "umap", group.by = "seurat_clusters",label = T)
DimPlot(seob_ATAC_PBMC, reduction = "umap", group.by = "celltype",label = T)


#**SEACells-------------------------------------------------------------------------------------
seurat2anndata(obj = seob_PBMC, 
               outFile = "/mnt/data/home/tycloud/workspace/algorithms_raw/data/seob_RNA_PBMC.h5ad", 
               slot = 'count', main_layer = 'RNA', 
               transfer_layers = NULL, 
               drop_single_values = FALSE)

names(seob_ATAC_PBMC@reductions)[1] = 'svd'
seurat2anndata(obj = seob_ATAC_PBMC, 
               outFile = "/mnt/data/home/tycloud/workspace/algorithms_raw/data/seob_ATAC_PBMC.h5ad", 
               slot = 'count', main_layer = 'peaks', 
               transfer_layers = NULL, 
               drop_single_values = FALSE)

#**DEpeak---------------------------------------------------------------------------------------
seacell = anndata2seurat(anndata_file = "/mnt/data/home/tycloud/workspace/algorithms_raw/data/seob_RNA_PBMC_seacells_600.h5ad")
seacell = seacell@meta.data
seacell$celltype = seob_PBMC@meta.data[rownames(seacell), 'celltype']

#为每个seacell分配最大的celltype
seacell_summary <- seacell %>%
  group_by(SEACell, celltype) %>%
  summarise(freq = n(), .groups = "drop") %>%
  arrange(SEACell, desc(freq))

seacell_most_common <- seacell_summary %>%
  group_by(SEACell) %>%
  slice_max(order_by = freq, n = 1) %>%
  ungroup()
seacell_most_common = as.data.frame(seacell_most_common)
seacell_most_common = seacell_most_common%>%
  distinct(SEACell, .keep_all = T)
rownames(seacell_most_common) = seacell_most_common$SEACell

#利用SEACell进行差异表达分析
seob_ATAC_PBMC_seacells = anndata2seurat(anndata_file = "/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_seacells_ad.h5ad")
seob_ATAC_PBMC_seacells = seob_ATAC_PBMC_seacells@assays$RNA@counts
rownames(seob_ATAC_PBMC_seacells) = gsub(':','-',rownames(seob_ATAC_PBMC_seacells))
comid = intersect(rownames(seob_ATAC_PBMC_seacells),
                  rownames(seob_ATAC_PBMC))
seob_ATAC_PBMC_seacells = seob_ATAC_PBMC_seacells[comid, ]
seob_ATAC_PBMC_seacells = CreateSeuratObject(counts = seob_ATAC_PBMC_seacells,
                                             meta.data = seacell_most_common)

Idents(seob_ATAC_PBMC_seacells) = seob_ATAC_PBMC_seacells@meta.data$celltype
CNCC_unknown = FindAllMarkers(seob_ATAC_PBMC_seacells,
                              Mandn.pct = 0.3,
                              only.pos=T,
                              logfc.threshold = 0.6)
CNCC_unknown_df_ATAC = CNCC_unknown
CNCC_unknown_df = CNCC_unknown_df_ATAC

#CNCC_unknown_df1 = CNCC_unknown_df[(CNCC_unknown_df$pct.1-CNCC_unknown_df$pct.2)>0.2, ]
CNCC_unknown_df$de_pct = (CNCC_unknown_df$pct.1-CNCC_unknown_df$pct.2)/CNCC_unknown_df$pct.1

CNCC_unknown_df = CNCC_unknown_df[CNCC_unknown_df$de_pct>0.5, ]
table(CNCC_unknown_df$cluster)

CNCC_unknown_df_ATAC_500 <- CNCC_unknown_df %>%
  group_by(cluster) %>%
  arrange(desc(avg_log2FC), .by_group = TRUE) %>%
  slice_head(n = 200) %>% # 每组取前200行
  ungroup()

CNCC_unknown_df_ATAC_500 = CNCC_unknown_df_ATAC_500%>%
  distinct(gene, .keep_all = T)

table(CNCC_unknown_df_ATAC_500$cluster)

CNCC_unknown_df_ATAC_3000 <- CNCC_unknown_df %>%
  group_by(cluster) %>%
  arrange(desc(avg_log2FC), .by_group = TRUE) %>%
  slice_head(n = 800) %>% # 每组取前200行
  ungroup()
CNCC_unknown_df_ATAC_3000 = CNCC_unknown_df_ATAC_3000%>%
  distinct(gene, .keep_all = T)



#**DEG--------------------------------------------------------------------------------
seob_PBMC_seacells = anndata2seurat(anndata_file = "/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_seacells_ad.h5ad")
seob_PBMC_seacells = seob_PBMC_seacells@assays$RNA@counts
comid = intersect(rownames(seob_PBMC_seacells),
                  rownames(seob_PBMC))
comid = intersect(comid, Protein_coding_Genes)
seob_PBMC_seacells = seob_PBMC_seacells[comid, ]
seob_PBMC_seacells = CreateSeuratObject(counts = seob_PBMC_seacells,
                                        meta.data = seacell_most_common)
Idents(seob_PBMC_seacells) = seob_PBMC_seacells@meta.data$celltype
CNCC_unknown = FindAllMarkers(seob_PBMC_seacells,
                              Mandn.pct = 0.2,
                              only.pos=T,
                              logfc.threshold = 0.2)

CNCC_unknown_df = CNCC_unknown
CNCC_unknown_df_RNA = CNCC_unknown_df
#CNCC_unknown_df1 = CNCC_unknown_df[(CNCC_unknown_df$pct.1-CNCC_unknown_df$pct.2)>0.2, ]
CNCC_unknown_df$de_pct = (CNCC_unknown_df$pct.1-CNCC_unknown_df$pct.2)/CNCC_unknown_df$pct.1
CNCC_unknown_df = CNCC_unknown_df[CNCC_unknown_df$avg_log2FC>0.3&
                                    CNCC_unknown_df$de_pct>0.3, ]
table(CNCC_unknown_df$cluster)

CNCC_unknown_df_RNA_100 <- CNCC_unknown_df %>%
  group_by(cluster) %>%
  arrange(desc(avg_log2FC), .by_group = TRUE) %>%
  slice_head(n = 100) %>% # 每组取前200行
  ungroup()
table(CNCC_unknown_df_RNA_100$cluster)

CNCC_unknown_df_RNA_100 = CNCC_unknown_df_RNA_100%>%
  distinct(gene, .keep_all = T)

CNCC_unknown_df_RNA_300 <- CNCC_unknown_df %>%
  group_by(cluster) %>%
  arrange(desc(avg_log2FC), .by_group = TRUE) %>%
  slice_head(n = 300) %>% # 每组取前200行
  ungroup()
table(CNCC_unknown_df_RNA_300$cluster)

CNCC_unknown_df_RNA_300 = CNCC_unknown_df_RNA_300%>%
  distinct(gene, .keep_all = T)

CNCC_unknown_df_RNA_300 = CNCC_unknown_df_RNA_300[!CNCC_unknown_df_RNA_300$gene%in%CNCC_unknown_df_RNA_100$gene, ]
CNCC_unknown_df_RNA_300$cluster = '-1'

CNCC_unknown_df_RNA_300 = CNCC_unknown_df_RNA_300[, c('gene','cluster')]
CNCC_unknown_df_RNA_100 = CNCC_unknown_df_RNA_100[, c('gene','cluster')]

node_RNA_DE = rbind(CNCC_unknown_df_RNA_100, CNCC_unknown_df_RNA_300)

CNCC_unknown_df_ATAC_3000 = CNCC_unknown_df_ATAC_3000[!CNCC_unknown_df_ATAC_3000$gene%in%CNCC_unknown_df_ATAC_500$gene, ]
CNCC_unknown_df_ATAC_3000$cluster = '-1'

CNCC_unknown_df_ATAC_3000 = CNCC_unknown_df_ATAC_3000[, c('gene','cluster')]
CNCC_unknown_df_ATAC_500 = CNCC_unknown_df_ATAC_500[, c('gene','cluster')]

node_ATAC_DE = rbind(CNCC_unknown_df_ATAC_3000, CNCC_unknown_df_ATAC_500)

#**HVG_ATAC------------------------------------------------------------------------------
seob_ATAC_PBMC_seacells = FindVariableFeatures(seob_ATAC_PBMC_seacells, nfeatures = 15000)
HVpeak = VariableFeatures(seob_ATAC_PBMC_seacells)

HVpeak = readRDS('~/workspace/algorithms_raw/data/PBMC_HVpeak.rds')

intersect(HVpeak, node_ATAC_DE$gene)
HVpeak = setdiff(HVpeak, node_ATAC_DE$gene)
HVpeak = data.frame(gene=HVpeak,
                    cluster=rep('-1', length(HVpeak)))
node_ATAC = rbind(node_ATAC_DE, HVpeak)

#**HVG_RNA------------------------------------------------------------------------------
seob_PBMC_seacells = FindVariableFeatures(seob_PBMC_seacells, nfeatures = 3000)
HVG = VariableFeatures(seob_PBMC_seacells)
intersect(HVG, node_RNA_DE$gene)
HVG = setdiff(HVG, node_RNA_DE$gene)
HVG = data.frame(gene=HVG,
                 cluster=rep('-1', length(HVG)))
node_RNA = rbind(node_RNA_DE, HVG)


#**generate_GRN---------------------------------------------------------------------------
library(JASPAR2024)
library(TFBSTools)
peaks = readRDS(file = '~/workspace/algorithms_raw/data/peaks_PBMC.rds')
Genes = readRDS(file = '~/workspace/algorithms_raw/data/genes_PBMC.rds')

peaks=peaks$gene
Genes=Genes$gene

#using ENCODE, ChIP-Altas and 4D_nucleome_project database
#all 
HiC_list = readRDS(file = "~/database/Encode/Encode_HiC_bed_data_list.rds")
TF_list = readRDS(file = "~/database/Encode/Encode_TF_bed_data_list.rds")
TF_list_BM = readRDS(file = "~/database/Encode/Encode_TF_Bone_marrow_bed_data_list.rds")
TF_list_ChIP_Altas = readRDS(file = "~/database/ChIP_Altas/ChIP-Altas.rds")

meta_data_TF_all = fread("/mnt/data/home/tycloud/database/Encode/TF_metadata.tsv",data.table = F)
meta_data_TF_all = meta_data_TF_all[meta_data_TF_all$`Biosample organism`=='Homo sapiens', ]
meta_data_TF_all = meta_data_TF_all[meta_data_TF_all$`File type`=='bed', ]

meta_data_HiC_all = fread("/mnt/data/home/tycloud/database/Encode/HiC_metadata.tsv",data.table = F)
meta_data_HiC_all = meta_data_HiC_all[meta_data_HiC_all$`Biosample organism`=='Homo sapiens', ]
meta_data_HiC_all = meta_data_HiC_all[meta_data_HiC_all$`Output type`%in%c('contact domains',
                                                                           'loops'), ]

meta_data_ChIP = read.table(file = '~/database/ChIP_Altas/experimentList.tab', 
                            header = F, 
                            sep = "\t", 
                            stringsAsFactors = F,
                            fill = T)
meta_data_ChIP = meta_data_ChIP[meta_data_ChIP$V2%in%c('hg38','hg19'),]
meta_data_ChIP = meta_data_ChIP[meta_data_ChIP$V3=='TFs and others', ]
meta_data_ChIP = meta_data_ChIP[!duplicated(meta_data_ChIP$V1), ]
rownames(meta_data_ChIP) = meta_data_ChIP$V1

#HiC dataset
#BM 
HiC_list_BM = readRDS(file = "~/database/Encode/Encode_HiC_Bone_marrow_bed_data_list.rds")

#K562
# HiC_list_4D_K562 = readRDS(file = "~/database/4D_nucleome_project/4D_nucleome_project_K562.rds")
# HiC_list_4D_K562 <- HiC_list_4D_K562[!names(HiC_list_4D_K562) %in% "4DNFIJU5XBK7.hic_loops"]
# 
# 
# #HCT116
# HiC_list_4D_HCT116 = readRDS(file = "~/database/4D_nucleome_project/4D_nucleome_project_HCT116.rds")
# 
# #hela
# HiC_list_4D_hela = readRDS(file = "~/database/4D_nucleome_project/4D_nucleome_project_hela.rds")
# 
# #A549
# HiC_list_4D_A549 = readRDS(file = "~/database/4D_nucleome_project/4D_nucleome_project_A549.rds")
# 
# #GM12878
# HiC_list_4D_GM12878 = readRDS(file = "~/database/4D_nucleome_project/4D_nucleome_project_GM12878.rds")
# 
# 
# HiC_list = c(HiC_list, HiC_list_BM, HiC_list_4D_K562,
#              HiC_list_4D_HCT116, HiC_list_4D_hela, 
#              HiC_list_4D_A549, HiC_list_4D_GM12878)

HiC_list = c(HiC_list, HiC_list_BM)



#select all Genes
#BM Encoder
TF_names = names(TF_list_BM)
TF_factors <- sapply(TF_names, function(x) strsplit(x, "_")[[1]][2])
TF_factors = toupper(TF_factors)
Genes = unique(c(Genes, TF_factors))



# generate GRN
GRN_ENCODE = generate_GRN_from_ENCODE(HiC_list=HiC_list,
                                      TF_list=TF_list,
                                      TF=TF,
                                      ppi=ppi,
                                      peaks=peaks,
                                      Genes=Genes,
                                      min_overlap=200,
                                      peak_genes=T,
                                      upstream=25000,
                                      downstream=25000,
                                      ncores=38,
                                      gtf=EnsDb.Hsapiens.v86,
                                      CRE_genes_df_tmp = '~/workspace/algorithms_raw/tmp/CRE_genes_df_PBMC.rds')
GRN_ENCODE_list =  lapply(GRN_ENCODE, function(x){colnames(x)=c('from','to');return(x)})
GRN_ENCODE_df = do.call('rbind', GRN_ENCODE_list)

#ChIP-Altas_4D_nucleome_project
GRN_ChIP_4D = generate_GRN_from_ENCODE(HiC_list=HiC_list,
                                       TF_list=TF_list_ChIP_Altas,
                                       TF=TF,
                                       ppi=ppi,
                                       peaks=peaks,
                                       Genes=Genes,
                                       min_overlap=200,
                                       peak_genes=T,
                                       upstream=25000,
                                       downstream=25000,
                                       ncores=38,
                                       gtf=EnsDb.Hsapiens.v86,
                                       CRE_genes_df_tmp = '~/workspace/algorithms_raw/tmp/CRE_genes_df_PBMC.rds')
GRN_ChIP_4D_list =  lapply(GRN_ChIP_4D, function(x){colnames(x)=c('from','to');return(x)})
GRN_ChIP_4D_list[['CRE_CRE_df']] = NULL
GRN_ChIP_4D_df = do.call('rbind', GRN_ChIP_4D_list)

GRN_df = rbind(GRN_ENCODE_df, GRN_ChIP_4D_df)
GRN_df <- unique(GRN_df, by = c("from", "to"))

#add BM in Encode 
GRN_BM = generate_GRN_from_ENCODE(HiC_list=HiC_list_BM,
                                  TF_list=TF_list_BM,
                                  TF=TF,
                                  ppi = ppi,
                                  peaks=peaks,
                                  Genes=Genes,
                                  min_overlap=200,
                                  peak_genes=T,
                                  upstream=25000,
                                  downstream=25000,
                                  ncores=1,
                                  gtf=EnsDb.Hsapiens.v86,
                                  CRE_genes_df_tmp = '~/workspace/algorithms_raw/tmp/CRE_genes_df_PBMC.rds')

GRN_BM_list =  lapply(GRN_BM, function(x){colnames(x)=c('from','to');return(x)})
GRN_BM_list = GRN_BM_list[c('TF_df_all','CRE_CRE_df')]
GRN_BM_df = do.call('rbind', GRN_BM_list)
GRN_BM_df = as.data.table(GRN_BM_df)

GRN_df = rbind(GRN_df, GRN_BM_df)
GRN_df <- unique(GRN_df, by = c("from", "to"))

peaks_df = data.frame(id=peaks,
                      idtype='CRE')
Genes_df = data.frame(id=Genes,
                      idtype='Target')

Genes_Peaks_df = rbind(Genes_df, peaks_df)
Genes_Peaks_df$idtype = case_when(Genes_Peaks_df$id%in%c(TF)~'TF',
                                  .default = as.character(Genes_Peaks_df$idtype))
Genes_Peaks_type_id = Genes_Peaks_df$idtype
names(Genes_Peaks_type_id) = Genes_Peaks_df$id


GRN_df[, edge_id:=paste(from,to,sep='_')]
GRN_df[, edge_id_type:=paste(unname(Genes_Peaks_type_id[GRN_df$from]),
                             unname(Genes_Peaks_type_id[GRN_df$to]),
                             sep='_')]
table(GRN_df$edge_id_type)
GRN_df = GRN_df[!edge_id_type%in%c('NA_NA','NA_Target',
                                   'NA_TF','Target_NA')]
GRN_df = na.omit(GRN_df)

GRN_df <- unique(GRN_df, by = c("from", "to"))

saveRDS(GRN_df, file = "~/workspace/algorithms_raw/data/PBMC_priorNetwork.rds")

#**genrate node feature data-------------------------------------------------------------------
GRN_df = readRDS(file = "~/workspace/algorithms_raw/data/PBMC_priorNetwork.rds")
seob_common_seacells = anndata2seurat(anndata_file = "/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_seacells_ad.h5ad")
common_seacells_mat = seob_common_seacells@assays$RNA@counts
rownames(common_seacells_mat) = gsub(":", "-", rownames(common_seacells_mat))
common_seacells_mat = as.data.frame(common_seacells_mat)

#save data
com_id = Reduce('intersect', list(rownames(common_seacells_mat),
                                  Genes_Peaks_df$id,
                                  unique(c(GRN_df$from, GRN_df$to))))


Genes_Peaks_df = Genes_Peaks_df[Genes_Peaks_df$id%in%com_id,]
GRN_df = GRN_df[GRN_df$from%in%com_id & GRN_df$to%in%com_id,  ]
common_seacells_mat = common_seacells_mat[com_id, ]

colnames(Genes_Peaks_df)=c('name','type')

write.csv(Genes_Peaks_df, file = "data/PBMC/raw_600/graph_0_node_names.csv")
write.csv(common_seacells_mat, file = "data/PBMC/raw_600/graph_0_nodes.csv")
write.csv(GRN_df, file = "data/PBMC/raw_600/graph_0_edges.csv",row.names = F)


#**genrate predicted GRN-----------------------------------------------------------------------
#
common_seacells_mat = fread(file = "data/PBMC/raw_600/graph_0_nodes.csv")%>%
  column_to_rownames(var = 'V1')


GRN_df = read.csv(file = "data/PBMC/raw_600/graph_0_edges.csv")
GRN_df = as.data.table(GRN_df)
Genes_Peaks_df = fread(file = "data/PBMC/raw_600/graph_0_node_names.csv",
                       header = T)%>%
  column_to_rownames(var = 'V1')
rownames(Genes_Peaks_df) = Genes_Peaks_df$name

#CRE_CRE
#pearson
GRN_df_CRE_CRE = GRN_df[GRN_df$edge_id_type=='CRE_CRE', ]
result <- calculate_fast_correlation(GRN = GRN_df_CRE_CRE, GET = common_seacells_mat, threshold = 0)
GRN_df_CRE_CRE = result; rm(result); gc()
GRN_df_CRE_CRE = GRN_df_CRE_CRE[abs(GRN_df_CRE_CRE$correlation)>0.5, ]
table(GRN_df_CRE_CRE$edge_id_T)

GRN_df_CRE_CRE = GRN_df_CRE_CRE%>%
  left_join(conns, by = c('edge_id'='edge_id'))
GRN_df_CRE_CRE = GRN_df_CRE_CRE[abs(GRN_df_CRE_CRE$coaccess)>0.9, ]
table(GRN_df_CRE_CRE$edge_id_T)


#利用eQTL确定长距离作用
CRE_genes_df_PBMC = readRDS('data/CRE_genes_df_PBMC.rds')

common_genes <- intersect(eQTL_SNP$Gene, CRE_genes_df_PBMC$Gene)
eQTL_SNP = eQTL_SNP[eQTL_SNP$Gene%in%common_genes, ]

eQTL_CRE_merge <- merge(
  eQTL_SNP, 
  CRE_genes_df_PBMC, 
  by = "Gene",
  allow.cartesian=TRUE
)

granges_CRE = unique(c(GRN_df_CRE_CRE$from, GRN_df_CRE_CRE$to))
granges_SNP = unique(eQTL_CRE_merge$CRE_eQTL)

ov = peaks_overlap(granges_SNP, granges_CRE, min_overlap = 1)
ov$peak_src = str_replace(ov$peak_src, ':', '-')
ov$peak_ref = str_replace(ov$peak_ref, ':', '-')
ov = ov%>%distinct(peak_src, .keep_all = T)

ov = ov[, .(CRE_eQTL=peak_src, CRE_re_qQTL=peak_ref)]

eQTL_CRE_merge <- merge(
  eQTL_CRE_merge, 
  ov, 
  by = "CRE_eQTL",
  allow.cartesian=TRUE
)

eQTL_CRE_merge = eQTL_CRE_merge[, .(from=CRE_re_qQTL,to=CRE)]
eQTL_CRE_merge[, edge_id:=paste(from,to,sep='_')]
eQTL_CRE_merge = eQTL_CRE_merge%>%distinct(edge_id, .keep_all = T)


#转为无向图
eQTL_CRE_merge_ = data.table(from=eQTL_CRE_merge$to,
                             to=eQTL_CRE_merge$from)
eQTL_CRE_merge_[, edge_id:=paste(from,to,sep='_')]

eQTL_CRE_merge= rbind(eQTL_CRE_merge, eQTL_CRE_merge_)

GRN_df_CRE_CRE$edge_id_T = ifelse(GRN_df_CRE_CRE$edge_id%in%eQTL_CRE_merge$edge_id,1,0)

result <- calculate_fast_correlation(GRN = GRN_df_CRE_CRE, GET = common_seacells_mat, threshold = 0)
GRN_df_CRE_CRE = result; rm(result); gc()
GRN_df_CRE_CRE_ = GRN_df_CRE_CRE[abs(GRN_df_CRE_CRE$correlation)>0.5, ]
table(GRN_df_CRE_CRE_$edge_id_T)
table(GRN_df_CRE_CRE$edge_id_T)

GRN_df_HC = c(GRN_df_TF_CRE$edge_id,
              GRN_df_CRE_CRE_$edge_id)


GRN_df_T = c(GRN_df_TF_CRE$edge_id,
             GRN_df_CRE_CRE[edge_id_T==1, .(edge_id)]$edge_id)


# select only TF CRE Target
GRN_df = GRN_df[GRN_df$edge_id_type%in%c('CRE_CRE','CRE_Target',
                                         'CRE_TF', 'TF_CRE'), ]

GRN_df$edge_id_HC = ifelse(GRN_df$edge_id%in%GRN_df_HC,1,0)
GRN_df$edge_id_T = ifelse(GRN_df$edge_id%in%GRN_df_T,1,0)
table(GRN_df$edge_id_type, GRN_df$edge_id_T)
table(GRN_df$edge_id_type, GRN_df$edge_id_HC)

TF_PBMC = Genes_Peaks_df[Genes_Peaks_df$type=='TF', 'name']
TF_TF = TF_Target[from%in%TF_PBMC & to%in%TF_PBMC]
TF_TF$edge_id = paste(TF_TF$from, TF_TF$to, sep = '_')
TF_TF$edge_id_type = 'TF_TF'
TF_TF$edge_id_T = 0
TF_TF$edge_id_HC = 0

GRN_df = rbind(GRN_df, TF_TF)

com_id = Reduce('intersect', list(rownames(common_seacells_mat),
                                  Genes_Peaks_df$name,
                                  unique(c(GRN_df$from, GRN_df$to))))
Genes_Peaks_df = Genes_Peaks_df[Genes_Peaks_df$name%in%com_id,]
GRN_df = GRN_df[GRN_df$from%in%com_id & GRN_df$to%in%com_id,  ]
common_seacells_mat = common_seacells_mat[com_id, ]

CRE_genes_df_PBMC = readRDS('~/workspace/algorithms_raw/tmp/CRE_genes_df_PBMC.rds')
Genes_Peaks_df$cluster = ifelse(Genes_Peaks_df$name%in%CRE_genes_df_PBMC$Gene,
                                1,0)


write.csv(Genes_Peaks_df, file = "/data_input/PBMC/raw_600/graph_0_node_names.csv")
write.csv(common_seacells_mat, file = "data_input/PBMC/raw_600/graph_0_nodes.csv")
write.csv(GRN_df, file = "data_input/PBMC/raw_600/graph_0_edges.csv",row.names = F)

table(GRN_df$edge_id_type)
table(GRN_df$edge_id_T,GRN_df$edge_id_type)
table(Genes_Peaks_df$type)






