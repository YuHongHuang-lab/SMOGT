# RunSeurat 
#' A Wrapper of seurat integrated, reduce and cluster
#'
#' @param seob  seurat object
#' @param split_by 
#' @param coln_add 
#' @param genes_add 
#' @param integrated 
#' @param integrated_assay 
#' @param cellid 
#' @param nfeatures 
#' @param npcs 
#' @param resolution 
#' @param k_anchor 
#' @param k_filter Adjusting the value of k.filter can change 
#' the number of cells considered as potential anchors. 
#' Higher k.filter values imply stricter screening criteria, 
#' which may reduce the number of anchors 
#' but improve the specificity of integration; 
#' lower k.filter values, on the other hand, 
#' may increase the number of anchors, 
#' which helps to capture a larger population of cells 
#' but may increase false integrations
#' @param k_score The k.score parameter defines 
#' the number of nearest neighbors to be used 
#' when evaluating the quality of each anchor candidate. 
#' This parameter is mainly used to evaluate the stability of 
#' each anchor point:
#' For each potential anchor point, FindIntegrationAnchors calculates its average similarity score with its k.score nearest neighbors.
#' This score is used as a measure of the consistency and reliability
#'  of the anchor point, with a high score 
#'  indicating that its neighbors have a high degree of 
#'  consistency across datasets.
#' 
#' @param features 
#' @param min_batch_num 
#' @param k_weight 
#' @param use_old_RNA 
#' @param use_old_Seurat 
#'
#' @return seurat object
#' @export
#'
#' @examples seob_E10 = RunSeurat(seob = seob_E10, 
#split_by="sample",
#integrated="no",
#cellid=NULL,
#nfeatures=5000,
#npcs=30,
#resolution=1,
#k_anchor=5,
#k_weight = 100,
#use_old_RNA = FALSE,
#use_old_Seurat = FALSE)
RunSeurat = function(seob, split_by=NULL,
                     coln_add=NULL,
                     cellid_add=T,
                     genes_add=NULL,
                     integrated=NULL,
                     integrated_assay="RNA",
                     cellid=NULL,
                     min_cells=0,
                     min_features=0,
                     max_features=0,
                     MT_filter = 0,
                     nfeatures=3000,
                     npcs=30,
                     resolution=0.3,
                     k_anchor=5,
                     k_filter=200,
                     k_score=30,
                     outfeatures=NULL,
                     min_batch_num=NULL,
                     k_weight=100,
                     use_old_RNA=F,
                     use_old_Seurat=F,
                     use_old_embeddings=F,
                     regress_CC=F,
                     genome_="mouse"){
  if (genome_=="mouse"){
    g2m.features = homologene::homologene(cc.genes$g2m.genes, inTax = 9606, outTax = 10090)
    g2m.features = g2m.features[, 2]
    s.features = homologene::homologene(cc.genes$s.genes, inTax = 9606, outTax = 10090)
    s.features = s.features[, 2]
  }
  if (isFALSE(use_old_Seurat)){
    if (class(seob)=="list"){
      colid = lapply(seob, function(x)colnames(x@meta.data))
      comcolid = Reduce("intersect", colid)
      seob = lapply(seob, function(x){x@meta.data=x@meta.data[, comcolid];x})
      seob <- mapply(function(x, name) {
        x@meta.data[, 'splitid'] = name;x
      }, seob, names(seob), SIMPLIFY = FALSE)
      split_order = names(seob)
      split_by = 'splitid'
      if (cellid_add){
        seob = merge(x=seob[[1]],y=seob[-1], 
                     add.cell.ids=names(seob))
      }else {
        seob = merge(x=seob[[1]],y=seob[-1])
      }
      seob@meta.data$splitid = factor(seob@meta.data$splitid,
                                      levels = split_order)
      DefaultAssay(seob) = "RNA"
    }
    DefaultAssay(seob) = "RNA"
    
    seob = CreateSeuratObject(counts = seob@assays$RNA@counts,
                              meta.data = seob@meta.data,
                              min.cells = min_cells,
                              min.features = min_features)
    
    if (MT_filter>0){
      seob[["percent.mt"]] <- PercentageFeatureSet(seob, pattern = "(^MT-)|(^mt-)")
      seob <- subset(seob, subset = percent.mt < MT_filter)
    }
    
    if (max_features>0){
      seob <- subset(seob, subset = nFeature_RNA  < max_features)
    }
    
    if (!is.null(cellid)){
      seob = subset(seob, cells=cellid)
    }
    
    if (!is.null(outfeatures)){
      outfeatures_ = grep("(Rik$)|(^Gm)|(^ENS)", rownames(seob@assays$RNA@counts), value = T)
      outfeatures = unique(c(outfeatures, outfeatures_))
      features = setdiff(rownames(seob@assays$RNA@counts), outfeatures)
      seob = subset(seob, features=features)
    }
    
    
    
    seob_old = seob
    
    if (integrated=="RPCA"){
      if (integrated_assay == "RNA"){
        if (is.null(coln_add)){
          seob = CreateSeuratObject(counts = seob@assays$RNA@counts,
                                    meta.data = seob@meta.data[, split_by, drop=FALSE])
        }else if (coln_add=="all"){
          seob = CreateSeuratObject(counts = seob@assays$RNA@counts,
                                    meta.data = seob@meta.data)
        }else {
          seob = CreateSeuratObject(counts = seob@assays$RNA@counts,
                                    meta.data = seob@meta.data[, c(split_by, coln_add), drop=FALSE])
        }
        if (!is.null(genes_add)){genes_add = intersect(rownames(seob), genes_add)}
        seob_list = SplitObject(seob,
                                split.by = split_by)
        
        if (!is.null(min_batch_num)){
          seob_list = lapply(seob_list, FUN = function(x){
            if (ncol(x)>min_batch_num){
              return(x)
            }else {
              return(NULL)
            }
          })
          seob_list = seob_list[!sapply(seob_list, is.null)]
        }
        
        
        seob_list <- lapply(X = seob_list, FUN = function(x) {
          x <- NormalizeData(x)
          x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = nfeatures)
        })
        
        features <- SelectIntegrationFeatures(object.list = seob_list,
                                              nfeatures = nfeatures)
        if (!is.null(genes_add)){features = unique(features, genes_add)}
        
        
        seob_list <- lapply(X = seob_list, FUN = function(x) {
          if (isTRUE(regress_CC)){
            x <- CellCycleScoring(x, 
                                  s.features = s.features, 
                                  g2m.features = g2m.features, 
                                  set.ident = TRUE)
            x <- ScaleData(x, vars.to.regress = c("S.Score", "G2M.Score"),verbose = T)
          }else {
            x <- ScaleData(x, features = features, verbose = TRUE)
          }
          
          x <- RunPCA(x, features = features, npcs = npcs, verbose = TRUE)
        })
        
        
        anchors <- FindIntegrationAnchors(object.list = seob_list, 
                                          anchor.features = features, 
                                          reduction = "rpca",
                                          k.anchor = k_anchor,
                                          k.filter = k_filter,
                                          k.score = k_score,
                                          dims = 1:npcs)
        seob <- IntegrateData(anchorset = anchors,
                              k.weight = k_weight)
        rm(anchors);gc()
        
        DefaultAssay(seob) <- "integrated"
        if (isTRUE(regress_CC)){
          seob <- ScaleData(seob, vars.to.regress = c("S.Score", "G2M.Score","nFeature_RNA"),verbose = T)
        }else {
          seob <- ScaleData(seob, verbose = TRUE)
        }
        # seob <- ScaleData(seob, verbose = TRUE)
        seob_old = subset(seob_old, cells=colnames(seob))
        
        if (isTRUE(use_old_RNA)){
          seob[["RNA"]] = seob_old[["RNA"]]
        }else {
          DefaultAssay(seob) = "RNA"
          seob <- NormalizeData(seob)
        }
        DefaultAssay(seob) <- "integrated"
      }else if (integrated_assay == "integrated"){
        if (is.null(coln_add)){
          seob = CreateSeuratObject(counts = seob@assays$integrated@data,
                                    meta.data = seob@meta.data[, split_by, drop=FALSE])
        }else if (coln_add=="all"){
          seob = CreateSeuratObject(counts = seob@assays$integrated@data,
                                    meta.data = seob@meta.data)
        }else {
          seob = CreateSeuratObject(counts = seob@assays$integrated@data,
                                    meta.data = seob@meta.data[, c(split_by, coln_add), drop=FALSE])
        }
        if (!is.null(genes_add)){genes_add = intersect(rownames(seob), genes_add)}
        seob_list = SplitObject(seob,
                                split.by = split_by)
        
        if (!is.null(min_batch_num)){
          seob_list = lapply(seob_list, FUN = function(x){
            if (ncol(x)>min_batch_num){
              return(x)
            }else {
              return(NULL)
            }
          })
          seob_list = seob_list[!sapply(seob_list, is.null)]
        }
        
        
        seob_list <- lapply(X = seob_list, FUN = function(x) {
          x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = nfeatures)
        })
        
        features <- SelectIntegrationFeatures(object.list = seob_list,
                                              nfeatures = nfeatures)
        if (!is.null(genes_add)){features = unique(features, genes_add)}
        
        
        seob_list <- lapply(X = seob_list, FUN = function(x) {
          x <- ScaleData(x, features = features, verbose = TRUE)
          x <- RunPCA(x, features = features, npcs = npcs, verbose = TRUE)
        })
        
        
        anchors <- FindIntegrationAnchors(object.list = seob_list, 
                                          anchor.features = features, 
                                          reduction = "rpca",
                                          k.anchor = k_anchor,
                                          k.filter = k_filter,
                                          k.score = k_score,
                                          dims = 1:npcs)
        seob <- IntegrateData(anchorset = anchors,
                              k.weight = k_weight)
        rm(anchors);gc()
        
        DefaultAssay(seob) <- "integrated"
        seob <- ScaleData(seob, verbose = TRUE)
        seob_old = subset(seob_old, cells=colnames(seob))
        
        if (isTRUE(use_old_RNA)){
          seob[["RNA"]] = seob_old[["RNA"]]
        }else {
          DefaultAssay(seob) = "RNA"
          seob <- NormalizeData(seob)
        }
        DefaultAssay(seob) <- "integrated"
      }
    }
    else if (integrated=="SCT"){
      if (is.null(coln_add)){
        seob = CreateSeuratObject(counts = seob@assays$RNA@counts,
                                  meta.data = seob@meta.data[, split_by, drop=FALSE])
      }else if (coln_add=="all"){
        seob = CreateSeuratObject(counts = seob@assays$RNA@counts,
                                  meta.data = seob@meta.data)
      }else {
        seob = CreateSeuratObject(counts = seob@assays$RNA@counts,
                                  meta.data = seob@meta.data[, c(split_by, coln_add), drop=FALSE])
      }
      if (!is.null(genes_add)){genes_add = intersect(rownames(seob), genes_add)}
      DefaultAssay(seob) = "RNA"
      seob_list = SplitObject(seob,
                              split.by = split_by)
      
      if (!is.null(min_batch_num)){
        seob_list = lapply(seob_list, FUN = function(x){
          if (ncol(x)>min_batch_num){
            return(x)
          }else {
            return(NULL)
          }
        })
        seob_list = seob_list[!sapply(seob_list, is.null)]
      }
      
      seob_list <- lapply(X = seob_list, FUN = function(x) {
        
        if (isTRUE(regress_CC)){
          x <- CellCycleScoring(x, 
                                s.features = s.features, 
                                g2m.features = g2m.features, 
                                set.ident = TRUE)
          x <- SCTransform(x,
                           variable.features.n = nfeatures,
                           vars.to.regress = c("S.Score", "G2M.Score"),
                           verbose = T)
        }else {
          x <- SCTransform(x,
                           variable.features.n = nfeatures,
                           verbose = T)
        }
      })
      
      features = SelectIntegrationFeatures(object.list = seob_list,
                                           nfeatures = nfeatures)
      if (!is.null(genes_add)){features = unique(features, genes_add)}
      seob_list = PrepSCTIntegration(object.list = seob_list,
                                     anchor.features = features)
      anchors = FindIntegrationAnchors(object.list = seob_list,
                                       normalization.method = "SCT",
                                       anchor.features = features,
                                       k.anchor = k_anchor,
                                       k.filter = k_filter,
                                       k.score = k_score,)
      seob = IntegrateData(anchorset = anchors,
                           normalization.method = "SCT",
                           dims = 1:npcs,
                           k.weight = k_weight)
      rm(anchors);gc()
      
      DefaultAssay(seob) <- "integrated"
      #seob <- ScaleData(seob, verbose = TRUE)
    }
    else if (integrated=="no"){
      if (is.null(coln_add)){
        seob = CreateSeuratObject(counts = seob@assays$RNA@counts,
                                  meta.data = seob@meta.data[, split_by, drop=FALSE])
      }else if (coln_add=="all"){
        seob = CreateSeuratObject(counts = seob@assays$RNA@counts,
                                  meta.data = seob@meta.data)
      }else {
        seob = CreateSeuratObject(counts = seob@assays$RNA@counts,
                                  meta.data = seob@meta.data[, c(split_by, coln_add), drop=FALSE])
      }
      
      DefaultAssay(seob) = "RNA"
      if (!is.null(genes_add)){genes_add = intersect(rownames(seob), genes_add)}
      seob <- NormalizeData(seob)
      seob <- FindVariableFeatures(seob, selection.method = "vst", nfeatures = nfeatures)
      seob@assays$RNA@var.features = unique(c(seob@assays$RNA@var.features, genes_add))
      if (isTRUE(regress_CC)){
        seob <- CellCycleScoring(seob, 
                                 s.features = s.features, 
                                 g2m.features = g2m.features, 
                                 set.ident = TRUE)
        seob <- ScaleData(seob, vars.to.regress = c("S.Score", "G2M.Score"),verbose = T)
      }else {
        seob <- ScaleData(seob, verbose = TRUE)
      }
    }
  }else {
    #seob_E12_Mand_Chai <- Seurat::UpdateSeuratObject(object = Mouse1stArchE12)
    if (!is.null(cellid)){
      seob = subset(seob, cells=cellid)
    }
    
    if (!is.null(outfeatures)){
      features = setdiff(rownames(seob@assays$RNA@counts), outfeatures)
      seob = seob[features, ]
    }
  }
  
  tryCatch({
    if (isTRUE(use_old_Seurat))DefaultAssay(seob)="integrated" 
  },
  error=function(m)DefaultAssay(seob)="RNA")
  if (isFALSE(use_old_embeddings)){
    seob <- RunPCA(seob, npcs = npcs, verbose = TRUE)
    seob <- RunUMAP(seob, reduction = "pca", dims = 1:npcs,
                    return.model = TRUE)
  }
  
  seob <- FindNeighbors(seob, reduction = "pca", dims = 1:npcs)
  seob <- FindClusters(seob, resolution = resolution)
  DefaultAssay(seob) = "RNA"
  if (integrated=="SCT"){
    seob = NormalizeData(seob)
  }
  rm(seob_old);gc()
  return(seob)
}

#' Title seurat2anndata convert seurat to adata
#'
#' @param obj 
#' @param outFile 
#' @param slot 
#' @param main_layer 
#' @param transfer_layers 
#' @param drop_single_values 
#' @param uns 
#' @param obsm_add 
#' @param obsm_add_name 
#'
#' @return
#' @export
#'
#' @examples bm_integrated_E11_E14 = seurat2anndata(obj = bm_integrated_E11_E14, 
#outFile = "data/E12_E14_WT_MT_CNCC_scVelo_scRNA_seq_E11_E14.h5ad", 
#slot = 'counts', main_layer = 'RNA', 
#transfer_layers = c("spliced", "unspliced", "ambiguous"), 
#drop_single_values = FALSE)
seurat2anndata <- function(
    obj,
    src=NULL,
    outFile = NULL, 
    slot = 'counts', 
    main_layer = 'RNA', 
    transfer_layers = c("spliced", "unspliced", "ambiguous"), 
    drop_single_values = TRUE,
    uns = NULL,
    obsm_add = NULL,
    obsm_add_name = NULL
) {
  
  if (!is.null(src)){
    com_id = intersect(colnames(obj), colnames(src))
    com_genes = intersect(rownames(obj), rownames(src))
    
    obj = obj[com_genes, com_id]
    src = src[com_genes, com_id]
    
    for (layer in transfer_layers) {
      obj[[layer]] = CreateAssayObject(counts = src@assays[[layer]]@counts)
    }
    
    rm(src); gc()
  }
  
  if (compareVersion(as.character(obj@version), '3.0.0') < 0)
    obj <- Seurat::UpdateSeuratObject(object = obj)
  
  X <- tryCatch({
    Seurat::GetAssayData(object = obj, assay = main_layer, slot = slot)
  },
  error=function(m){
    message(m)
    return(obj[[main_layer]][slot])
  })
  
  
  obs <- .regularise_df(obj@meta.data, drop_single_values = drop_single_values)
  
  var <- .regularise_df(Seurat::GetAssay(obj, assay = main_layer)@meta.features, drop_single_values = drop_single_values)
  
  obsm <- NULL
  reductions <- names(obj@reductions)
  if (length(reductions) > 0) {
    obsm <- sapply(
      reductions,
      function(name) as.matrix(Seurat::Embeddings(obj, reduction=name)),
      simplify = FALSE
    )
    names(obsm) <- paste0('X_', tolower(names(obj@reductions)))
  }
  
  if (!is.null(obsm_add)){
    obsm[[obsm_add_name]] = obsm_add
  }
  
  layers <- list()
  for (layer in transfer_layers) {
    mat <- Seurat::GetAssayData(object = obj, assay = layer, slot = slot)
    layers[[layer]] <- Matrix::t(mat)
  }
  
  anndata <- reticulate::import('anndata', convert = FALSE)
  
  adata <- anndata$AnnData(
    X = Matrix::t(X),
    obs = obs,
    var = var,
    obsm = obsm,
    layers = layers,
    uns = uns
  )
  
  if (!is.null(outFile))
    adata$write_h5ad(outFile)
  
  adata
}

#' Title seurat2anndata convert data to seurat
#'
#' @param anndata_file 
#' @param outFile 
#'
#' @return
#' @export
#'
#' @examples
#' seacell = anndata2seurat(anndata_file = "/mnt/data/home/tycloud/workspace/algorithms_raw/data/seob_RNA_A549_seacells.h5ad")
#' 
#' 
anndata2seurat <- function(
    anndata_file,
    outFile = NULL
) {
  library(reticulate)
  library(Seurat)
  library(Matrix)
  
  # Import necessary Python modules
  anndata <- import('anndata')
  sc <- import('scanpy')
  
  # Read the AnnData file
  adata <- sc$read_h5ad(anndata_file)
  
  # Create Seurat object
  expr_matrix <- t(adata$X)
  colnames(expr_matrix) <- adata$obs_names$tolist()
  rownames(expr_matrix) <- adata$var_names$tolist()
  
  if (ncol(as.data.frame(adata$obs))==0){
    seurat_object <- CreateSeuratObject(counts = expr_matrix)
  }else {
    seurat_object <- CreateSeuratObject(counts = expr_matrix,
                                        meta.data = as.data.frame(adata$obs))
  }
  
  # seurat_object <- CreateSeuratObject(counts = expr_matrix,
  #                                     meta.data = as.data.frame(adata$obs)) 
  
  # Add feature data
  main_layer <- 'RNA'
  var <- as.data.frame(adata$var)
  if (ncol(var)!=0){
    seurat_object@assays[[main_layer]]@meta.features <- var
  }
  # seurat_object@assays[[main_layer]]@meta.features <- var
  
  # Add dimensionality reduction results
  for (reduction_name in adata$obsm_keys()) {
    reduction_matrix <- as.matrix(adata$obsm[[reduction_name]])
    rownames(reduction_matrix) <- adata$obs_names$tolist()
    reduction_key <- gsub("X_", "", reduction_name)
    colnames(reduction_matrix) = toupper(paste0(paste0(reduction_key, "_"), 1:ncol(reduction_matrix)))
    seurat_object[[reduction_key]] <- CreateDimReducObject(
      embeddings = reduction_matrix, key = toupper(paste0(reduction_key, "_")), assay = main_layer
    )
  }
  
  # Optionally save the Seurat object
  if (!is.null(outFile)) {
    saveRDS(seurat_object, file = outFile)
  }
  
  return(seurat_object)
}



function(HiC_list=NULL,
         TF_list=NULL,
         TF=NULL,
         ppi=NULL,
         peaks,
         Genes,
         min_overlap=200,
         peak_genes=T,
         upstream=1000,
         downstream=1000,
         ncores=16,
         gtf=NULL,
         HiC_model='Contact',
         Contact_min_overlap=600,
         CRE_genes_df_tmp='tmp/CRE_genes_df.rds'){
  
  #Gene-Gene
  TF_ = intersect(Genes, TF)
  result = list()
  if (!is.null(TF) & !is.null(ppi)){
    TF_ = intersect(Genes, TF)
    TF_TF_df = ppi[from %in% TF_ & to %in% TF_]
    
    Target_Target_df = ppi[from %in% Genes & to %in% Genes]
    
    TF_TF_df[, edge_id := paste(from, to, sep = "_")]
    Target_Target_df[, edge_id := paste(from, to, sep = "_")]    
    
    Target_Target_df <- Target_Target_df[!edge_id %in% TF_TF_df$edge_id]
    
    Target_Target_df[, edge_id := NULL]
    Target_Target_df = Target_Target_df[!from%in%TF_]
    
    TF_TF_df[, edge_id := NULL]
    
    result[['TF_TF_df']]=TF_TF_df
    result[['Target_Target_df']]=Target_Target_df
  }
  
  # peak_Gene
  if (isTRUE(peak_genes)){
    if (!file.exists(CRE_genes_df_tmp)){
      CRE_genes_df = find_peak_genes(gtf=gtf,
                                     peaks=peaks,
                                     Genes=Genes,
                                     upstream=upstream,
                                     downstream=downstream,
                                     min_overlap=min_overlap,
                                     ncores=ncores)
      CRE_genes_df = do.call(rbind, CRE_genes_df)
      saveRDS(CRE_genes_df, file = CRE_genes_df_tmp)
    }else {
      CRE_genes_df = readRDS(file = CRE_genes_df_tmp)
    }
    result[['CRE_genes_df']] = CRE_genes_df
    Gene_CRE = unique(CRE_genes_df$CRE)
  }else {
    Gene_CRE=peaks
  }
  # CRE_genes_df = readRDS(file = 'tmp/CRE_genes_df.rds')
  
  #CRE_CRE
  if (!is.null(HiC_list)){
    if (HiC_model=='Contact'){
      if (ncores==1){
        CRE_CRE_df = data.frame()
        for (name in names(HiC_list)){
          print(name)
          HiC_ = HiC_list[[name]]
          HiC_ = data.frame(peak1=str_c(HiC_$V1,HiC_$V2,HiC_$V3,sep = '-'),
                            peak2=str_c(HiC_$V4,HiC_$V5,HiC_$V6,sep = '-'))
          peaks_overlap_1 = peaks_overlap(peaks = peaks,
                                          peaks_ref = HiC_$peak1,
                                          min_overlap = min_overlap)
          peaks_overlap_1$peak_src = str_replace( peaks_overlap_1$peak_src,':','-')
          peaks_overlap_1$peak_ref = str_replace( peaks_overlap_1$peak_ref,':','-')
          
          peaks_overlap_ <- left_join(HiC_, peaks_overlap_1, by = c('peak1'='peak_ref'))
          colnames(peaks_overlap_)[length(peaks_overlap_)] = 'peak_a'
          
          peaks_overlap_2 = peaks_overlap(peaks = peaks,
                                          peaks_ref = HiC_$peak2,
                                          min_overlap = min_overlap)
          peaks_overlap_2$peak_src = str_replace( peaks_overlap_2$peak_src,':','-')
          peaks_overlap_2$peak_ref = str_replace( peaks_overlap_2$peak_ref,':','-')
          
          peaks_overlap_ <- left_join(peaks_overlap_, peaks_overlap_2, by = c('peak2'='peak_ref'))
          colnames(peaks_overlap_)[length(peaks_overlap_)] = 'peak_b'
          
          peaks_overlap_ = na.omit(peaks_overlap_)
          peaks_overlap_ = peaks_overlap_[, c('peak_a','peak_b')]
          
          
          
          CRE_CRE_df = rbind(CRE_CRE_df, peaks_overlap_)
        }
        CRE_CRE_df = as.data.table(CRE_CRE_df)
      }
      else {
        process_HiC <- function(name, 
                                HiC_list, 
                                peaks, 
                                min_overlap) {
          HiC_ <- HiC_list[[name]]
          HiC_ <- data.frame(peak1 = str_c(HiC_$V1, HiC_$V2, HiC_$V3, sep = '-'),
                             peak2 = str_c(HiC_$V4, HiC_$V5, HiC_$V6, sep = '-'))
          
          peaks_overlap_1 <- peaks_overlap(peaks = peaks,
                                           peaks_ref = HiC_$peak1,
                                           min_overlap = min_overlap)
          peaks_overlap_1$peak_src <- str_replace(peaks_overlap_1$peak_src, ':', '-')
          peaks_overlap_1$peak_ref <- str_replace(peaks_overlap_1$peak_ref, ':', '-')
          
          peaks_overlap_ <- left_join(HiC_, peaks_overlap_1, by = c('peak1' = 'peak_ref'))
          colnames(peaks_overlap_)[length(peaks_overlap_)] <- 'peak_a'
          
          peaks_overlap_2 <- peaks_overlap(peaks = peaks,
                                           peaks_ref = HiC_$peak2,
                                           min_overlap = min_overlap)
          peaks_overlap_2$peak_src <- str_replace(peaks_overlap_2$peak_src, ':', '-')
          peaks_overlap_2$peak_ref <- str_replace(peaks_overlap_2$peak_ref, ':', '-')
          
          peaks_overlap_ <- left_join(peaks_overlap_, peaks_overlap_2, by = c('peak2' = 'peak_ref'))
          colnames(peaks_overlap_)[length(peaks_overlap_)] <- 'peak_b'
          
          peaks_overlap_ <- na.omit(peaks_overlap_)
          peaks_overlap_ <- peaks_overlap_[, c('peak_a', 'peak_b')]
          peaks_overlap_ = as.data.table(peaks_overlap_)
          
          return(peaks_overlap_)
        }
        CRE_CRE_df <- mclapply(names(HiC_list), 
                               process_HiC, 
                               HiC_list = HiC_list, 
                               peaks = peaks, 
                               min_overlap = min_overlap, 
                               mc.cores = ncores)
        # CRE_CRE_df = data.table()
        # for (df in CRE_CRE_df_){
        #   CRE_CRE_df = rbind(CRE_CRE_df, df)
        #   # CRE_CRE_df = CRE_CRE_df[unique(CRE_CRE_df$peak_a, CRE_CRE_df$peak_b), ]
        #   CRE_CRE_df <- unique(CRE_CRE_df, by = c("peak_a", "peak_b"))
        # }
        
        CRE_CRE_df = do.call(rbind, CRE_CRE_df)
        CRE_CRE_df = as.data.table(CRE_CRE_df)
        CRE_CRE_df <- unique(CRE_CRE_df, by = c("peak_a", "peak_b"))
      }
    }
    else if (HiC_model=='Boundaries'){
      if (ncores==1){
        HiC_ = HiC_list[[name]]
        # HiC_ = HiC_[HiC_$V4=='Strong',]
        HiC_ = data.frame(peak1=str_c(HiC_$V1,
                                      HiC_$V2-30000,
                                      HiC_$V3+30000,sep = '-'))
        
        peaks_overlap_df = peaks_overlap(peaks = peaks,
                                         peaks_ref = HiC_$peak1,
                                         min_overlap = Contact_min_overlap)
        peaks_overlap_df$peak_src <- str_replace(peaks_overlap_df$peak_src, ':', '-')
        peaks_overlap_df$peak_ref <- str_replace(peaks_overlap_df$peak_ref, ':', '-')
        TAD_groups <- split(peaks_overlap_df$peak_src, peaks_overlap_df$peak_ref)
        paired_peaks_list = lapply(TAD_groups, function(group){
          n <- length(group)
          if (n > 1) {
            # 使用 combn 函数生成所有两两组合
            pairs <- combn(group, 2)
            data.table(from = pairs[1, ], to = pairs[2, ], stringsAsFactors = FALSE)
          } else {
            NULL
          }
        })
        paired_peaks_df = do.call('rbind',paired_peaks_list)
      }
      else {
        process_HiC <- function(name, 
                                HiC_list, 
                                peaks, 
                                Contact_min_overlap) {
          HiC_ <- HiC_list[[name]]
          # HiC_ = HiC_[HiC_$V4=='Strong',]
          HiC_ = data.frame(peak1=str_c(HiC_$V1,
                                        HiC_$V2-30000,
                                        HiC_$V3+30000,sep = '-'))        
          peaks_overlap_df = peaks_overlap(peaks = peaks,
                                           peaks_ref = HiC_$peak1,
                                           min_overlap = Contact_min_overlap)
          peaks_overlap_df$peak_src <- str_replace(peaks_overlap_df$peak_src, ':', '-')
          peaks_overlap_df$peak_ref <- str_replace(peaks_overlap_df$peak_ref, ':', '-')
          TAD_groups <- split(peaks_overlap_df$peak_src, peaks_overlap_df$peak_ref)
          paired_peaks_list = lapply(TAD_groups, function(group){
            n <- length(group)
            if (n > 1) {
              # 使用 combn 函数生成所有两两组合
              pairs <- combn(group, 2)
              data.table(from = pairs[1, ], to = pairs[2, ], stringsAsFactors = FALSE)
            } else {
              NULL
            }
          })
          paired_peaks_df = do.call('rbind',paired_peaks_list)
          
          return(paired_peaks_df)
        }
        CRE_CRE_df <- mclapply(names(HiC_list), 
                               process_HiC, 
                               HiC_list = HiC_list, 
                               peaks = peaks, 
                               Contact_min_overlap = Contact_min_overlap, 
                               mc.cores = ncores)
        CRE_CRE_df = do.call(rbind, CRE_CRE_df)
        colnames(CRE_CRE_df) = c("peak_a", "peak_b")
        CRE_CRE_df <- unique(CRE_CRE_df, by = c("peak_a", "peak_b"))
      }
    }
    CRE_CRE_df = CRE_CRE_df[peak_a %in% Gene_CRE | peak_b %in% Gene_CRE]
    result[['CRE_CRE_df']] = CRE_CRE_df
    CRE_CRE = unique(c(CRE_CRE_df$peak_a, CRE_CRE_df$peak_b))
  }else {
    CRE_CRE = peaks
  }
  
  #TF_CRE
  TF_names = names(TF_list)
  TF_factors <- sapply(TF_names, function(x) strsplit(x, "_")[[1]][2])
  TF_factors = toupper(TF_factors)
  TF_list <- TF_list[TF_factors %in% TF_]
  
  if (ncores==1){
    TF_df_all = data.table()
    for (name in names(TF_list)){
      print(name)
      TF_new = TF_list[[name]]
      TF_new = data.frame(TF_new)
      TF_new = str_c(TF_new$seqnames, TF_new$start, TF_new$end, sep = '-')
      peaks_overlap_ = peaks_overlap(peaks = peaks,
                                     peaks_ref = TF_new,
                                     min_overlap = min_overlap)
      peaks_overlap_$peak_src = str_replace(peaks_overlap_$peak_src,':','-')
      peaks_overlap_$peak_ref = str_replace(peaks_overlap_$peak_ref,':','-')
      TF_df = data.table(TF=rep(str_split(name, '_', simplify = T)[,2],length(unique(peaks_overlap_$peak_src))),
                         CRE=unique(peaks_overlap_$peak_src))
      TF_df_all = rbind(TF_df_all, TF_df)
    }
  }
  else {
    process_TF <- function(name, TF_list, peaks, min_overlap) {
      TF_ <- TF_list[[name]]
      TF_ <- as.data.frame(TF_)
      TF_ <- str_c(TF_$seqnames, TF_$start, TF_$end, sep = '-')
      
      peaks_overlap_ <- peaks_overlap(peaks = peaks,
                                      peaks_ref = TF_,
                                      min_overlap = min_overlap)
      peaks_overlap_$peak_src <- str_replace(peaks_overlap_$peak_src, ':', '-')
      peaks_overlap_$peak_ref <- str_replace(peaks_overlap_$peak_ref, ':', '-')
      
      TF_df <- data.table(TF = rep(str_split(name, '_', simplify = TRUE)[, 2], 
                                   length(unique(peaks_overlap_$peak_src))),
                          CRE = unique(peaks_overlap_$peak_src))
      
      return(TF_df)
    }
    TF_df_all <- mclapply(names(TF_list), 
                          process_TF, 
                          TF_list = TF_list, 
                          peaks = peaks, 
                          min_overlap = min_overlap, 
                          mc.cores = ncores)
    
    # TF_df_all = data.table()
    # for (df in TF_df_all_){
    #   TF_df_all = rbind(TF_df_all, df)
    #   # CRE_CRE_df = CRE_CRE_df[unique(CRE_CRE_df$peak_a, CRE_CRE_df$peak_b), ]
    #   TF_df_all <- unique(TF_df_all, by = c("TF", "CRE"))
    # }
    TF_df_all = do.call(rbind, TF_df_all)
    TF_df_all <- unique(TF_df_all, by = c("TF", "CRE"))
    # TF_df_all = TF_df_all[TF_df_all$TF%in%Genes, ]
  }
  
  TF_df_all$TF = toupper(TF_df_all$TF)
  TF_df_all = TF_df_all[CRE%in%CRE_CRE]
  TF_df_all = TF_df_all[TF%in%TF_]
  result[['TF_df_all']] = TF_df_all
  
  return(result)
}


calculate_fast_correlation <- function(GRN, GET, method = "pearson", threshold = NULL) {
  # 确保输入的表格中包含 "from" 和 "to" 列
  if (!("from" %in% colnames(GRN)) || !("to" %in% colnames(GRN))) {
    stop("GRN表格必须包含 'from' 和 'to' 列")
  }
  
  # 检查GET矩阵是否包含GRN中指定的基因
  unique_genes <- unique(c(GRN$from, GRN$to))
  com_genes <- intersect(unique_genes, rownames(GET))
  GRN = GRN[GRN$from%in%com_genes & GRN$to%in%com_genes, ]
  GET = GET[com_genes, ]
  
  expr_matrix <- GET[com_genes, ]
  
  if (method == "pearson") {
    n <- ncol(expr_matrix)
    expr_centered <- t(scale(t(expr_matrix), center = TRUE, scale = TRUE))
    
    correlations <- numeric(nrow(GRN))
    
    from_expr <- expr_centered[GRN$from, ]
    to_expr <- expr_centered[GRN$to, ]
    
    correlations <- rowSums(from_expr * to_expr) / n
  } 
  else if(method=='cosine'){
    gene_norms <- sqrt(rowSums(expr_matrix^2))
    
    gene_norms[gene_norms < 1e-8] <- 1e-8
    
    expr_normalized <- expr_matrix / gene_norms
    
    from_expr <- expr_normalized[GRN$from, ]
    to_expr <- expr_normalized[GRN$to, ]
    
    correlations <- rowSums(from_expr * to_expr)
  }
  else if (method=='dot'){
    from_expr <- expr_matrix[GRN$from, ]
    to_expr <- expr_matrix[GRN$to, ]
    
    dot_products <- rowSums(from_expr * to_expr)
    
    correlations <- 1 / (1 + exp(-dot_products))
  }
  
  if (!is.null(threshold)) {
    mask <- abs(correlations) > threshold
    GRN <- GRN[mask, ]
    correlations <- correlations[mask]
  }
  
  GRN$correlation <- correlations
  gc()
  return(GRN)
}
