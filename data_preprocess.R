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


cell_marker_all = readxl::read_xlsx("../scATAC/data/Cell_marker_All.xlsx")
cell_marker_all = cell_marker_all[cell_marker_all$species=="Human", ]

TF_Target = readRDS(file = "~/database/TF_Target/TF_Target_Human.rds")

TF_Target = rbind(ppi, TF_Target)
TF_Target <- unique(TF_Target, by = c("from", "to"))

#*load data------------------------------------------------------------------------
seob_BM = readRDS(file = "~/workspace/algorithms_raw/data/HSPC.rds")
seob_ATAC_BM = readRDS(file = "~/workspace/algorithms_raw/data/HSPC_ATAC_BM.rds")

seob_LSC = readRDS(file = "~/workspace/algorithms_raw/data/LSC.rds")
seob_ATAC_LSC = readRDS(file = "~/workspace/algorithms_raw/data/ATAC_LSC.rds")

seob_TALL = readRDS(file = "~/workspace/algorithms_raw/data/TALL.rds")
seob_ATAC_TALL = readRDS(file = "~/workspace/algorithms_raw/data/ATAC_TALL.rds")

seob_MPAL = readRDS(file = "~/workspace/algorithms_raw/data/MPAL.rds")
seob_ATAC_MPAL = readRDS(file = "~/workspace/algorithms_raw/data/ATAC_MPAL.rds")

chain <- rtracklayer::import.chain("~/database/UCSC/hg19ToHg38.over.chain")

#*data process--------------------------------------------------------------------
#*---------------------------------------------------------------------
#*eQTL SNP---------------------------------------------------------------------
eQTL_SNP = fread(file = '~/database/GWAS/blood_eQTL/2019-12-11-cis-eQTLsFDR0.05-ProbeLevel-CohortInfoRemoved-BonferroniAdded.txt', data.table = T)
eQTL_SNP = eQTL_SNP[, .(SNP, SNPChr, SNPPos, GeneSymbol)]
eQTL_SNP_Trans = fread(file = '~/database/GWAS/blood_eQTL/2018-09-04-trans-eQTLsFDR0.05-CohortInfoRemoved-BonferroniAdded.txt.gz', data.table = T)
eQTL_SNP_Trans = eQTL_SNP_Trans[, .(SNP, SNPChr, SNPPos, GeneSymbol)]
eQTL_SNP = rbind(eQTL_SNP, eQTL_SNP_Trans)

gr_eQTL_hg19 <- GRanges(
  seqnames = paste('chr',eQTL_SNP$SNPChr,sep = ''),
  ranges   = IRanges(start = eQTL_SNP$SNPPos, end = eQTL_SNP$SNPPos),
  GeneSymbol = eQTL_SNP$GeneSymbol  # 可把其它注释信息放在 metadata 中
)

granges_peaks = liftOver(gr_eQTL_hg19, chain)
is_converted <- elementNROWS(granges_peaks) == 1
granges_peaks = unlist(granges_peaks[is_converted])
eQTL_SNP = eQTL_SNP[is_converted, ]
CRE <- paste0(seqnames(granges_peaks), "-", start(granges_peaks), "-", end(granges_peaks))
eQTL_SNP$CRE = CRE

eQTL_SNP = eQTL_SNP[,.(CRE_eQTL=CRE, Gene=GeneSymbol)]

#*---------------------------------------------------------------------
#*Persad et al.Heathy BM---------------------------------------------------------------------
#**snRNA--------------------------------------------------------------------------------------
#rep 1
counts <- Read10X_h5(filename = "data/raw/Persad_SEACells_BM/BM_1/BM_multiome_1/outs/filtered_feature_bc_matrix.h5")
counts = counts$`Gene Expression`
seob_1 = CreateSeuratObject(counts = counts)

#rep 2
counts <- Read10X_h5(filename = "data/raw/Persad_SEACells_BM/BM_2/BM_multiome_2/outs/filtered_feature_bc_matrix.h5")
counts = counts$`Gene Expression`
seob_2 = CreateSeuratObject(counts = counts)

seob_BM = RunSeurat(list('rep_1'=seob_1,
                         'rep_2'=seob_2), 
                    split_by=NULL,
                    coln_add=NULL,
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

seob_BM@meta.data$seurat_clusters = as.character(seob_BM@meta.data$seurat_clusters)

DimPlot(seob_BM, reduction = "umap", group.by = "seurat_clusters",label = T)

#get out of lymphocyte
cellid = rownames(seob_BM@meta.data[!seob_BM@meta.data$seurat_clusters%in%c('0','3','8','10'), ])
seob_BM = subset(seob_BM, cells=cellid)

seob_BM = RunSeurat(seob_BM, 
                    split_by=NULL,
                    coln_add=NULL,
                    cellid_add=T,
                    genes_add=NULL,
                    integrated="no",
                    integrated_assay="RNA",
                    cellid=NULL,
                    min_cells=3,
                    min_features=200,
                    max_features=0,
                    MT_filter = 20,
                    nfeatures=1500,
                    npcs=10,
                    resolution=0.6,
                    k_anchor=5,
                    k_filter=200,
                    k_score=30,
                    outfeatures=bad_genes,
                    min_batch_num=NULL,
                    k_weight=100)
DimPlot(seob_BM, reduction = "umap", group.by = "seurat_clusters",label = T)

seob_BM@meta.data$celltype = as.character(seob_BM@meta.data$seurat_clusters)
saveRDS(seob_BM, file = "data/HSPC.rds")

seob_BM = readRDS("data/HSPC.rds")

#**snATAC---------------------------------------------------------------------------------------
counts <- Read10X_h5(filename = "data/raw/Persad_SEACells_BM/BM_1/BM_multiome_1/outs/filtered_feature_bc_matrix.h5")
counts = counts$Peaks
metadata <- read.csv(
  file = "data/raw/Persad_SEACells_BM/BM_1/BM_multiome_1/outs/per_barcode_metrics.csv",
  header = TRUE,
  row.names = 1
)

chrom_assay <- CreateChromatinAssay(
  counts = counts,
  sep = c(":", "-"),
  fragments = 'data/raw/Persad_SEACells_BM/BM_1/BM_multiome_1/outs/atac_fragments.tsv.gz',
  min.cells = 10,
  min.features = 200
)

hspc_1 <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "peaks",
  meta.data = metadata
)

# extract gene annotations from EnsDb
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)

# change to UCSC style since the data was mapped to hg38
seqlevels(annotations) <- paste0('chr', seqlevels(annotations))
genome(annotations) <- "hg38"

Annotation(hspc_1) <- annotations

# compute nucleosome signal score per cell
hspc_1 <- NucleosomeSignal(object = hspc_1)

# compute TSS enrichment score per cell
hspc_1 <- TSSEnrichment(object = hspc_1, fast = T)

# add blacklist ratio and fraction of reads in peaks
hspc_1$atac_passed_filters = hspc_1$atac_raw_reads-
  hspc_1$atac_unmapped_reads - hspc_1$atac_lowmapq-
  hspc_1$atac_dup_reads - hspc_1$atac_chimeric_reads - 
  hspc_1$atac_mitochondrial_reads
hspc_1$atac_pct_reads_in_peaks <- (hspc_1$atac_peak_region_fragments / hspc_1$atac_passed_filters) * 100



hspc_1$high.tss <- ifelse(hspc_1$TSS.enrichment > 3, 'High', 'Low')
# TSSPlot(hspc_1, group.by = 'high.tss') + NoLegend()


hspc_1$nucleosome_group <- ifelse(hspc_1$nucleosome_signal > 4, 'NS > 4', 'NS < 4')
FragmentHistogram(object = hspc_1, group.by = 'nucleosome_group')

VlnPlot(
  object = hspc_1,
  features = c('nCount_peaks', 'TSS.enrichment', 'blacklist_ratio', 'nucleosome_signal', 'atac_pct_reads_in_peaks'),
  pt.size = 0.1,
  ncol = 5
)

hspc_1 <- subset(
  x = hspc_1,
  subset = nCount_peaks > 3000 &
    nCount_peaks < 30000 &
    atac_pct_reads_in_peaks > 15 &
    nucleosome_signal < 4 &
    TSS.enrichment > 1
)
hspc_1

#rep2
fragpath <- "data/raw/Persad_SEACells_BM/BM_2/BM_multiome_2/outs/atac_fragments.tsv.gz"
fragcounts <- CountFragments(fragments = fragpath)
atac.frags <- CreateFragmentObject(path = fragpath)
counts <- FeatureMatrix(
  fragments = atac.frags,
  features = granges(hspc_1)
)

atac.assay <- CreateChromatinAssay(
  counts = counts,
  fragments = atac.frags
)
hspc_2 <- CreateSeuratObject(counts = atac.assay, assay = "peaks")

hspc_1$sample = "rep_1"
hspc_2$sample = "rep_2"

seob_ATAC_list = list("rep_1"=hspc_1,
                      "rep_2"=hspc_2)
seob_ATAC_BM = merge(x = seob_ATAC_list[[1]],
                     y = seob_ATAC_list[-1],
                     add.cell.ids=names(seob_ATAC_list))
rm(seob_ATAC_list);gc()
#select DE & high variable peaks, genes in non T cells
seob_ATAC_BM = subset(seob_ATAC_BM, cells=colnames(seob_BM))
seob_ATAC_BM@meta.data$celltype = seob_BM@meta.data[rownames(seob_ATAC_BM@meta.data),
                                                    "celltype"]

seob_ATAC_BM <- RunTFIDF(seob_ATAC_BM)
seob_ATAC_BM <- FindTopFeatures(seob_ATAC_BM, min.cutoff = 'q75')
seob_ATAC_BM <- RunSVD(seob_ATAC_BM)

seob_ATAC_BM <- RunUMAP(object = seob_ATAC_BM, reduction = 'lsi', dims = 2:30)
seob_ATAC_BM <- FindNeighbors(object = seob_ATAC_BM, reduction = 'lsi', dims = 2:30)
seob_ATAC_BM <- FindClusters(object = seob_ATAC_BM, verbose = FALSE, algorithm = 3)

DimPlot(seob_ATAC_BM, reduction = "umap", group.by = "seurat_clusters",label = T)
DimPlot(seob_ATAC_BM, reduction = "umap", group.by = "celltype",label = T)

gene.activities <- GeneActivity(seob_ATAC_BM)
seob_ATAC_BM[['RNA']] <- CreateAssayObject(counts = gene.activities)

seob_ATAC_BM <- NormalizeData(
  object = seob_ATAC_BM,
  assay = 'RNA',
  normalization.method = 'LogNormalize',
  scale.factor = median(seob_ATAC_BM$nCount_RNA)
)


saveRDS(seob_ATAC_BM, file = "data/HSPC_ATAC_BM.rds")

#**SEACells-------------------------------------------------------------------------------------
seurat2anndata(obj = seob_BM, 
               outFile = "data/seob_RNA_BM.h5ad", 
               slot = 'count', main_layer = 'RNA', 
               transfer_layers = NULL, 
               drop_single_values = FALSE)

names(seob_ATAC_BM@reductions)[1] = 'svd'
seurat2anndata(obj = seob_ATAC_BM, 
               outFile = "data/seob_ATAC_BM.h5ad", 
               slot = 'count', main_layer = 'peaks', 
               transfer_layers = NULL, 
               drop_single_values = FALSE)


#**DEpeak---------------------------------------------------------------------------------------
seacell = anndata2seurat(anndata_file = "/mnt/data/home/tycloud/workspace/algorithms_raw/data/seob_RNA_BM_seacells.h5ad")
seacell = seacell@meta.data
seacell$celltype = seob_BM@meta.data[rownames(seacell), 'celltype']

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
seob_ATAC_BM_seacells = anndata2seurat(anndata_file = "/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_seacells_ad.h5ad")
seob_ATAC_BM_seacells = seob_ATAC_BM_seacells@assays$RNA@counts
rownames(seob_ATAC_BM_seacells) = gsub(':','-',rownames(seob_ATAC_BM_seacells))
comid = intersect(rownames(seob_ATAC_BM_seacells),
                  rownames(seob_ATAC_BM))
seob_ATAC_BM_seacells = seob_ATAC_BM_seacells[comid, ]
seob_ATAC_BM_seacells = CreateSeuratObject(counts = seob_ATAC_BM_seacells,
                                           meta.data = seacell_most_common)

Idents(seob_ATAC_BM_seacells) = seob_ATAC_BM_seacells@meta.data$celltype
CNCC_unknown = FindAllMarkers(seob_ATAC_BM_seacells,
                              Mandn.pct = 0.1,
                              only.pos=T,
                              logfc.threshold = 0.4)
CNCC_unknown_df_ATAC = CNCC_unknown
CNCC_unknown_df = CNCC_unknown_df_ATAC

#CNCC_unknown_df1 = CNCC_unknown_df[(CNCC_unknown_df$pct.1-CNCC_unknown_df$pct.2)>0.2, ]
CNCC_unknown_df$de_pct = (CNCC_unknown_df$pct.1-CNCC_unknown_df$pct.2)/CNCC_unknown_df$pct.1

CNCC_unknown_df = CNCC_unknown_df[CNCC_unknown_df$de_pct>0.5, ]
table(CNCC_unknown_df$cluster)

CNCC_unknown_df_ATAC_500 <- CNCC_unknown_df %>%
  group_by(cluster) %>%
  arrange(desc(avg_log2FC), .by_group = TRUE) %>%
  slice_head(n = 500) %>% # 每组取前200行
  ungroup()

CNCC_unknown_df_ATAC_500 = CNCC_unknown_df_ATAC_500%>%
  distinct(gene, .keep_all = T)

table(CNCC_unknown_df_ATAC_500$cluster)

CNCC_unknown_df_ATAC_3000 <- CNCC_unknown_df %>%
  group_by(cluster) %>%
  arrange(desc(avg_log2FC), .by_group = TRUE) %>%
  slice_head(n = 3000) %>% # 每组取前200行
  ungroup()
CNCC_unknown_df_ATAC_3000 = CNCC_unknown_df_ATAC_3000%>%
  distinct(gene, .keep_all = T)





saveRDS(peaks, file = "data/peaks.rds")
saveRDS(Genes, file = "data/Genes.rds")

# peaks = readRDS(file = "data/peaks.rds")


#**DEG--------------------------------------------------------------------------------
seob_BM_seacells = anndata2seurat(anndata_file = "/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_seacells_ad.h5ad")
seob_BM_seacells = seob_BM_seacells@assays$RNA@counts
comid = intersect(rownames(seob_BM_seacells),
                  rownames(seob_BM))
comid = intersect(comid, Protein_coding_Genes)
seob_BM_seacells = seob_BM_seacells[comid, ]
seob_BM_seacells = CreateSeuratObject(counts = seob_BM_seacells,
                                      meta.data = seacell_most_common)
Idents(seob_BM_seacells) = seob_BM_seacells@meta.data$celltype
CNCC_unknown = FindAllMarkers(seob_BM_seacells,
                              Mandn.pct = 0.2,
                              only.pos=T,
                              logfc.threshold = 0.2)

CNCC_unknown_df = CNCC_unknown
CNCC_unknown_df_RNA = CNCC_unknown_df
#CNCC_unknown_df1 = CNCC_unknown_df[(CNCC_unknown_df$pct.1-CNCC_unknown_df$pct.2)>0.2, ]
CNCC_unknown_df$de_pct = (CNCC_unknown_df$pct.1-CNCC_unknown_df$pct.2)/CNCC_unknown_df$pct.1
CNCC_unknown_df = CNCC_unknown_df[CNCC_unknown_df$avg_log2FC>0.5&
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
seob_ATAC_BM_seacells = FindVariableFeatures(seob_ATAC_BM_seacells, nfeatures = 15000)
HVpeak = VariableFeatures(seob_ATAC_BM_seacells)

intersect(HVpeak, node_ATAC_DE$gene)
HVpeak = setdiff(HVpeak, node_ATAC_DE$gene)
HVpeak = data.frame(gene=HVpeak,
                    cluster=rep('-1', length(HVpeak)))
node_ATAC = rbind(node_ATAC_DE, HVpeak)

#**HVG_RNA------------------------------------------------------------------------------
seob_BM_seacells = FindVariableFeatures(seob_BM_seacells, nfeatures = 2000)
HVG = VariableFeatures(seob_BM_seacells)
intersect(HVG, node_RNA_DE$gene)
HVG = setdiff(HVG, node_RNA_DE$gene)
HVG = data.frame(gene=HVG,
                 cluster=rep('-1', length(HVG)))
node_RNA = rbind(node_RNA_DE, HVG)


#**generate_GRN---------------------------------------------------------------------------
library(JASPAR2024)
library(TFBSTools)
peaks = node_ATAC$gene
Genes = node_RNA$gene

#using ENCODE, ChIP-Altas and 4D_nucleome_project database
#all 
HiC_list = readRDS(file = "~/database/Encode/Encode_HiC_bed_data_list.rds")
TF_list = readRDS(file = "~/database/Encode/Encode_TF_bed_data_list.rds")
HiC_list_BM = readRDS(file = "~/database/Encode/Encode_HiC_Bone_marrow_bed_data_list.rds")
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

#select all Genes
TF_Target_Human_BM_add = readRDS(file = '~/workspace/algorithms_raw/data/TF_Target_Human_BM_add.rds')
Target_Target_BM_add = readRDS(file = '~/workspace/algorithms_raw/data/Target_Target_BM_Human_BM_add.rds')


Genes = unique(c(Genes, 
                 TF_Target_Human_BM_add$from, 
                 TF_Target_Human_BM_add$to,
                 Target_Target_BM_add$from,
                 Target_Target_BM_add$to))
Genes = unique(toupper(Genes))

#2 BM Encoder
TF_names = names(TF_list_BM)
TF_factors <- sapply(TF_names, function(x) strsplit(x, "_")[[1]][2])
TF_factors = toupper(TF_factors)
Genes = unique(c(Genes, TF_factors))

#3 ChIP-Altas 
meta_data = read.table(file = '~/database/ChIP_Altas/experimentList.tab', 
                       header = F, 
                       sep = "\t", 
                       stringsAsFactors = F,
                       fill = T)
meta_data = meta_data[meta_data$V2%in%c('hg38','hg19'),]
meta_data = meta_data[meta_data$V3=='TFs and others', ]
meta_data = meta_data[!duplicated(meta_data$V1), ]
rownames(meta_data) = meta_data$V1
meta_data_ChIP = meta_data

TF_id = c('ERX181469','SRX3789540','SRX386202','SRX386203','SRX386204','SRX4385554','SRX5167211','SRX5167212',
          'SRX5186323','SRX5186323','SRX5254847','SRX5254848','SRX5254849','SRX5254850','SRX5254851',
          'SRX5254852','SRX5258180','SRX5567179','SRX5567180','SRX5574342','SRX5574343','SRX5574344','SRX5574345','SRX5574350','SRX658606',
          'SRX658607','SRX6763478','SRX687554','SRX687556',
          'SRX7626336','SRX8418557','SRX8418558',
          'SRX751541','SRX5409709','SRX3768736',
          'SRX386202','SRX386203','SRX386204',
          'SRX5409709','SRX646125','SRX646126','SRX646127',
          'SRX646129','SRX658606','SRX995497','SRX995496')
TF_id = unique(TF_id)

meta_data_ChIP_BM = meta_data_ChIP[TF_id,]
id = str_c(meta_data_ChIP_BM[, 'V1'],
           meta_data_ChIP_BM[, 'V4'],
           sep = '_')
intersect(id,names(TF_list_ChIP_Altas))
TF_list_ChIP_Altas_BM = TF_list_ChIP_Altas[id]

TF_names = names(TF_list_ChIP_Altas_BM)
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
                                      upstream=15000,
                                      downstream=15000,
                                      ncores=36,
                                      gtf=EnsDb.Hsapiens.v86,
                                      CRE_genes_df_tmp = '~/workspace/algorithms_raw/tmp/CRE_genes_df_BM.rds')
GRN_ENCODE_list =  lapply(GRN_ENCODE, function(x){colnames(x)=c('from','to');return(x)})
GRN_ENCODE_df = do.call('rbind', GRN_ENCODE_list)

#4D_nucleome_project
GRN_ChIP_4D = generate_GRN_from_ENCODE(HiC_list=HiC_list,
                                       TF_list=TF_list_ChIP_Altas,
                                       TF=TF,
                                       ppi=ppi,
                                       peaks=peaks,
                                       Genes=Genes,
                                       min_overlap=200,
                                       peak_genes=T,
                                       upstream=15000,
                                       downstream=15000,
                                       ncores=36,
                                       gtf=EnsDb.Hsapiens.v86,
                                       CRE_genes_df_tmp = '~/workspace/algorithms_raw/tmp/CRE_genes_df_BM.rds')
GRN_ChIP_4D_list =  lapply(GRN_ChIP_4D, function(x){colnames(x)=c('from','to');return(x)})
GRN_ChIP_4D_list[['CRE_CRE_df']] = NULL
GRN_ChIP_4D_df = do.call('rbind', GRN_ChIP_4D_list)

GRN_df = rbind(GRN_ENCODE_df, GRN_ChIP_4D_df)
GRN_df <- unique(GRN_df, by = c("from", "to"))

#add HSPC in Encode 
GRN_BM = generate_GRN_from_ENCODE(HiC_list=HiC_list_BM,
                                  TF_list=TF_list_BM,
                                  TF=TF,
                                  ppi = ppi,
                                  peaks=peaks,
                                  Genes=Genes,
                                  min_overlap=200,
                                  peak_genes=T,
                                  upstream=15000,
                                  downstream=15000,
                                  ncores=1,
                                  gtf=EnsDb.Hsapiens.v86,
                                  CRE_genes_df_tmp = '~/workspace/algorithms_raw/tmp/CRE_genes_df_BM.rds')

GRN_BM_list =  lapply(GRN_BM, function(x){colnames(x)=c('from','to');return(x)})
GRN_BM_list = GRN_BM_list[c('TF_df_all','CRE_CRE_df')]
GRN_BM_df = do.call('rbind', GRN_BM_list)
GRN_BM_df = as.data.table(GRN_BM_df)

GRN_df = rbind(GRN_df, GRN_BM_df)
GRN_df <- unique(GRN_df, by = c("from", "to"))
saveRDS(GRN_df, file = "data/BM_priorNetwork.rds")

#ChIP-Altas
GRN_ChIP_Altas_BM = generate_GRN_from_ENCODE(HiC_list=HiC_list,
                                             TF_list=TF_list_ChIP_Altas_BM,
                                             TF=TF,
                                             ppi=ppi,
                                             peaks=peaks,
                                             Genes=Genes,
                                             min_overlap=200,
                                             peak_genes=T,
                                             upstream=15000,
                                             downstream=15000,
                                             ncores=36,
                                             gtf=EnsDb.Hsapiens.v86,
                                             CRE_genes_df_tmp = '~/workspace/algorithms_raw/tmp/CRE_genes_df_BM.rds')
GRN_ChIP_Altas_BM_list =  lapply(GRN_ChIP_Altas_BM, function(x){colnames(x)=c('from','to');return(x)})
GRN_ChIP_Altas_BM_list = GRN_ChIP_Altas_BM_list[c('TF_df_all')]
GRN_ChIP_Altas_df_BM = do.call('rbind', GRN_ChIP_Altas_BM_list)

GRN_BM_df = rbind(GRN_BM_df, GRN_ChIP_Altas_df_BM)
GRN_BM_df <- unique(GRN_BM_df, by = c("from", "to"))

GRN_df = rbind(GRN_df, GRN_BM_df)
GRN_df <- unique(GRN_df, by = c("from", "to"))
saveRDS(GRN_df, file = "~/workspace/algorithms_raw/data/BM_priorNetwork.rds")

#pubmed BIOGRID add
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

#TF_TF
TF_TF_BM_add = TF_Target_Human_BM_add[from%in%TF & 
                                        to%in%TF]
TF_TF_BM_add[, edge_id:=paste(from,to,sep='_')]
TF_TF_BM_add[, edge_id_type:=paste(unname(Genes_Peaks_type_id[TF_TF_BM_add$from]),
                                   unname(Genes_Peaks_type_id[TF_TF_BM_add$to]),
                                   sep='_')]

#Target_Target
Target_Target_BM_add[, edge_id:=paste(from,to,sep='_')]
Target_Target_BM_add[, edge_id_type:=paste(unname(Genes_Peaks_type_id[Target_Target_BM_add$from]),
                                           unname(Genes_Peaks_type_id[Target_Target_BM_add$to]),
                                           sep='_')]
Target_Target_BM_add = na.omit(Target_Target_BM_add)


#TF CRE
TF_CRE_BM_add = convert_to_TF_CRE_Target(TF_Target = TF_Target_Human_BM_add, 
                                         CRE_Target = GRN_df[edge_id_type=='CRE_Target'],
                                         CRE_CRE = GRN_BM_list[['CRE_CRE_df']],
                                         TF_CRE = GRN_df[edge_id_type=='TF_CRE'])

TF_CRE_BM_add[, edge_id:=paste(from,to,sep='_')]
TF_CRE_BM_add[, edge_id_type:=paste(unname(Genes_Peaks_type_id[TF_CRE_BM_add$from]),
                                    unname(Genes_Peaks_type_id[TF_CRE_BM_add$to]),
                                    sep='_')]



GRN_BM_df_add = rbind(TF_TF_BM_add,
                      Target_Target_BM_add,
                      TF_CRE_BM_add)
#add with origin
GRN_BM_df[, edge_id:=paste(from,to,sep = '_')]
GRN_BM_df[, edge_id_type:=paste(unname(Genes_Peaks_type_id[GRN_BM_df$from]),
                                unname(Genes_Peaks_type_id[GRN_BM_df$to]),
                                sep='_')]
GRN_BM_df = rbind(GRN_BM_df, GRN_BM_df_add)
GRN_BM_df = unique(GRN_BM_df, by=c('from', 'to'))
saveRDS(GRN_BM_df, file = "~/workspace/algorithms_raw/data/BM_real_priorNetwork.rds")

GRN_df = rbind(GRN_df, GRN_BM_df)
GRN_df <- unique(GRN_df, by = c("from", "to"))
saveRDS(GRN_df, file = "~/workspace/algorithms_raw/data/BM_priorNetwork.rds")

GRN_df[,edge_id_T:=ifelse(edge_id%in%GRN_BM_df$edge_id,1,0)]

#**genrate node feature data-------------------------------------------------------------------
GRN_df = readRDS(file = "~/workspace/algorithms_raw/data/BM_priorNetwork.rds")
seob_common_seacells = anndata2seurat(anndata_file = "/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_seacells_ad.h5ad")
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
# colnames(GRN_df)[1:2]=c('src','dst')
GRN_df = GRN_df[, c(1,2,5)]

write.csv(Genes_Peaks_df, file = "~/workspace/algorithms_raw/data/HSPC/raw/graph_0_node_names.csv")
write.csv(common_seacells_mat, file = "~/workspace/algorithms_raw/data/HSPC/raw/graph_0_nodes.csv")
write.csv(GRN_df, file = "~/workspace/algorithms_raw/data/HSPC/raw/graph_0_edges.csv",row.names = F)


#**genrate predicted GRN 1000-----------------------------------------------------------------------
#cicero
common_seacells_mat = fread(file = "~/workspace/algorithms_raw/data/HSPC/raw/graph_0_nodes.csv",data.table = F)%>%
  column_to_rownames(var = 'V1')


GRN_df = read.csv(file = "~/workspace/algorithms_raw/data/HSPC/raw_1000/graph_0_edges.csv")
GRN_df = as.data.table(GRN_df)
Genes_Peaks_df = fread(file = "~/workspace/algorithms_raw/data/HSPC/raw_1000/graph_0_node_names.csv",
                       header = T)%>%
  column_to_rownames(var = 'V1')
rownames(Genes_Peaks_df) = Genes_Peaks_df$name

library(cicero)
ATAC_id = Genes_Peaks_df[Genes_Peaks_df$type=='CRE', 'name']

ATAC_mat = common_seacells_mat[ATAC_id, ]
ATAC_mat = t(ATAC_mat)

ATAC_mat = apply(ATAC_mat, 2, function(x){
  ifelse(x>median(x),1,0)
})
ATAC_mat = t(ATAC_mat)

cellinfo = data.frame(cellname=colnames(ATAC_mat),
                      row.names = colnames(ATAC_mat))
peakinfo = data.frame(site_name=str_replace(rownames(ATAC_mat),'(:)|(-)','_'))
rownames(peakinfo) = peakinfo$site_name

rownames(ATAC_mat) = peakinfo$site_name
colnames(ATAC_mat) = rownames(cellinfo)

input_cds = new_cell_data_set(ATAC_mat,
                              cell_metadata = cellinfo,
                              gene_metadata = peakinfo)
input_cds <- input_cds[Matrix::rowSums(exprs(input_cds)) != 0,] 
set.seed(2017)
input_cds <- detect_genes(input_cds)
input_cds <- estimate_size_factors(input_cds)
input_cds <- preprocess_cds(input_cds, method = "LSI")
input_cds <- reduce_dimension(input_cds, reduction_method = 'UMAP', 
                              preprocess_method = "LSI")

plot_cells(input_cds)
umap_coords <- reducedDims(input_cds)$UMAP
cicero_cds <- make_cicero_cds(input_cds, reduced_coordinates = umap_coords)

library(EnsDb.Hsapiens.v86)

# 提取染色体长度信息
edb <- EnsDb.Hsapiens.v86

# 获取基因组信息（染色体及长度）
genome_info <- seqinfo(edb)
chromosome_lengths <- as.data.frame(genome_info)

# 筛选人类常用染色体（1-22, X, Y）
chromosome_lengths <- chromosome_lengths[rownames(chromosome_lengths) %in% c(as.character(1:22), "X", "Y"), ]

# 格式化结果
chromosome_lengths <- data.frame(
  Chromosome = rownames(chromosome_lengths),
  Length = as.numeric(chromosome_lengths$seqlengths)
)
rownames(chromosome_lengths) <- NULL
colnames(chromosome_lengths) = c('V1', 'V2')
chromosome_lengths$V1 = str_c('chr', chromosome_lengths$V1)
chromosome_lengths$V2 = as.numeric(chromosome_lengths$V2)


conns <- run_cicero(cicero_cds, chromosome_lengths) 
saveRDS(conns, file = "~/workspace/algorithms_raw/data/HSPC/processed/Cicero.rds")
conns = readRDS(file = "~/workspace/algorithms_raw/data/HSPC/processed/Cicero.rds")

#TF_CRE
GRN_df_non_CRE_CRE = GRN_df[GRN_df$edge_id_type!='CRE_CRE', ]

result <- calculate_fast_correlation(GRN = GRN_df_non_CRE_CRE, GET = common_seacells_mat, threshold = 0)
GRN_df_non_CRE_CRE = result;rm(result); gc()

GRN_df_non_CRE_CRE$correlation = as.numeric(GRN_df_non_CRE_CRE$correlation)
GRN_df_non_CRE_CRE = GRN_df_non_CRE_CRE[GRN_df_non_CRE_CRE$correlation!=1, ]

GRN_df_TF_CRE = GRN_df_non_CRE_CRE[GRN_df_non_CRE_CRE$edge_id_type=='TF_CRE', ]
GRN_df_TF_CRE = GRN_df_TF_CRE[abs(GRN_df_TF_CRE$correlation)>0.3, ]
table(GRN_df_TF_CRE$edge_id_T)
table(GRN_df$edge_id_T, GRN_df$edge_id_type)

#CRE_CRE
#pearson
GRN_df_CRE_CRE = GRN_df[GRN_df$edge_id_type=='CRE_CRE', ]
result <- calculate_fast_correlation(GRN = GRN_df_CRE_CRE, GET = common_seacells_mat, threshold = 0)
GRN_df_CRE_CRE = result; rm(result); gc()
GRN_df_CRE_CRE = GRN_df_CRE_CRE[abs(GRN_df_CRE_CRE$correlation)>0.3, ]
table(GRN_df_CRE_CRE$edge_id_T)
table(GRN_df$edge_id_T, GRN_df$edge_id_type)

#cicero
conns$Peak1 = gsub('_','-',conns$Peak1)
conns$Peak2 = gsub('_','-',conns$Peak2)
conns$edge_id = paste(conns$Peak1,conns$Peak2,sep='_')
conns = conns[, c('edge_id', 'coaccess')]

GRN_df_CRE_CRE = GRN_df[GRN_df$edge_id_type=='CRE_CRE', ]
intersect(GRN_df_CRE_CRE$edge_id,
          conns$edge_id)
GRN_df_CRE_CRE = GRN_df_CRE_CRE%>%
  left_join(conns, by = c('edge_id'='edge_id'))

GRN_df_CRE_CRE = GRN_df_CRE_CRE[abs(GRN_df_CRE_CRE$coaccess)>0.4, ]
table(GRN_df_CRE_CRE$edge_id_T)
table(GRN_df$edge_id_T, GRN_df$edge_id_type)

GRN_df_HC = c(GRN_df_TF_CRE$edge_id,
              GRN_df_CRE_CRE$edge_id)

# select only TF CRE Target
GRN_df = GRN_df[GRN_df$edge_id_type%in%c('CRE_CRE','CRE_Target',
                                         'CRE_TF', 'TF_CRE'), ]

TF_BM = Genes_Peaks_df[Genes_Peaks_df$type=='TF', 'name']
TF_TF = TF_Target[from%in%TF_BM & to%in%TF_BM]
TF_TF$edge_id = paste(TF_TF$from, TF_TF$to, sep = '_')
TF_TF$edge_id_type = 'TF_TF'
TF_TF$edge_id_T = 0
TF_TF$edge_id_HC = 0

GRN_df = rbind(GRN_df, TF_TF)

GRN_df$edge_id_HC = ifelse(GRN_df$edge_id%in%GRN_df_HC,1,0)

com_id = Reduce('intersect', list(rownames(common_seacells_mat),
                                  Genes_Peaks_df$name,
                                  unique(c(GRN_df$from, GRN_df$to))))
Genes_Peaks_df = Genes_Peaks_df[Genes_Peaks_df$name%in%com_id,]
GRN_df = GRN_df[GRN_df$from%in%com_id & GRN_df$to%in%com_id,  ]
common_seacells_mat = common_seacells_mat[com_id, ]

CRE_genes_df_BM = readRDS('~/workspace/algorithms_raw/tmp/CRE_genes_df_BM.rds')
Genes_Peaks_df$cluster = ifelse(Genes_Peaks_df$name%in%CRE_genes_df_BM$Gene,
                                1,0)

write.csv(Genes_Peaks_df, file = "~/workspace/algorithms_raw/data/HSPC/raw_1000/graph_0_node_names.csv")
write.csv(common_seacells_mat, file = "~/workspace/algorithms_raw/data/HSPC/raw_1000/graph_0_nodes.csv")
write.csv(GRN_df, file = "~/workspace/algorithms_raw/data/HSPC/raw_1000/graph_0_edges.csv",row.names = F)

table(GRN_df$edge_id_type)
table(GRN_df$edge_id_T,GRN_df$edge_id_type)
table(GRN_df$edge_id_HC,GRN_df$edge_id_type)

GRN_df = read.csv(file = "~/workspace/algorithms_raw/data/HSPC/raw_1000/graph_0_edges.csv")
table(GRN_df$edge_id_type)
table(GRN_df$edge_id_T,GRN_df$edge_id_type)