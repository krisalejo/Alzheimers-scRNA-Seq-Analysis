import os
import anndata
import json
import numpy as np
from scipy.stats import zscore
from scipy.stats import pearsonr
import pandas as pd
from datetime import datetime
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import linregress
import os
import scipy.stats as stats
from shapely.geometry import Point, Polygon
import umap.umap_ as umap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
from matplotlib.colorbar import ColorbarBase
import matplotlib.colors as mcolors
import time
import tarfile
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from scipy import sparse
from matplotlib import cm
import json

class Tools:
    
    color_palette = {
        "0": "#07ba7f",
        "1": "#00FF00",
        "2": "#0000FF",
        "3": "#7B68EE",
        "4": "#00FFFF",
        "5": "#FFFF00",
        "6": "#FF00FF",
        "7": "#FF7A00",
        "8": "#008080",
        "9": "#800000",
        "10": "#008000",
        "11": "#000080",
        "12": "#D59269",
        "13": "#C0C0C0",
        "14": "#FFA500",
        "15": "#ADD8E6",
        "16": "#90EE90",
        "17": "#FFB6C1",
        "18": "#FFD700",
        "19": "#FF4500",
        "20": "#ADFF2F",
        "21": "#7B68EE",
        "22": "#00FA9A",
        "23": "#48D1CC",
        "24": "#C71585",
        "25": "#8B4513",
        "26": "#4682B4",
        "27": "#D2B48C",
        "28": "#D8BFD8",
        "29": "#FF6347",
        "30": "#F5DEB3"
    }
    
    color_palette_celltype = {
        "Cortical Neurons I": "#07ba7f", 
        "Cortical Neurons II": "#00FF00", 
        "Cortical Neurons III": "#0000FF", 
        "Cortical Neurons IV": "#7B68EE", 
        "GABAergic Neurons": "#00FFFF", 
        "Astrocytes I": "#FFFF00", 
        "VLMC": "#FF00FF", 
        "Astrocytes II": "#FF7A00", 
        "Oligodendrocytes": "#008080", 
        "Micro-PVM": "#800000", 
        "Neurons": "#008000", 
        "Striatal Neurons": "#000080", 
        "Hippocampal Neurons": "#D59269"
    }

    color_palette_celltype_filtered = {
        "Hippocampal Neurons": "#07ba7f",                       
        "Cortical Neurons I": "#00FF00", 
        "Cortical Neurons II": "#0000FF", 
        "Cortical Neurons III": "#7B68EE", 
        "VLMC I": "#00FFFF", 
        "VIP I": "#FFFF00", 
        "Pvalb": "#FF00FF", 
        "Cortical Neurons IV": "#FF7A00", 
        "Hypothalamic Neurons": "#008080", 
        "Endothelial Cells": "#800000", 
        "Cortical Neurons V": "#008000", 
        "Oligodendrocytes I": "#000080", 
        "Oligodendrocytes II": "#D59269", 
        "Micro-PVM": "#C0C0C0", 
        "VLMC II": "#FFA500", 
        "OPC": "#ADD8E6", 
        "GABAergic Neurons": "#90EE90", 
        "Striatal Neurons": "#FFB6C1", 
        "Cortical Neurons VII": "#FFD700", 
        "Astrocytes": "#FF4500", 
        "VIP II": "#ADFF2F", 
        "Neurons": "#7B68EE", 
        "Glutamatergic Neurons": "#00FA9A"
    }
    
    def __init__(self, adata = None, color_palette = None):
        self.adata = adata
        self.color_palette = color_palette
        

    def plot_umap(self, color = 'leiden', s = 1):
        plt.figure(figsize = (10,8))
        cluster_list = list(self.adata.obs[color].unique())
        for cluster in cluster_list:
            inds = self.adata.obs[color] == cluster
            x = self.adata.obsm['X_umap'][:,0][inds]
            y = self.adata.obsm['X_umap'][:,1][inds]
            palette = self.color_palette[cluster]
            plt.scatter(x, y, s=s, marker='.', c = palette)
            plt.title(f'{color} cluster umap with {self.adata.obs.shape[0]} cells')
            plt.axis("off")
        plt.show()

    def plot_brain_proj(self, color = 'leiden', s = 10):
        plt.figure(figsize = (10,8))
        cluster_list = list(self.adata.obs[color].unique())
        for cluster in cluster_list:
            inds = self.adata.obs[color] == cluster
            x = self.adata.obsm['X_spatial'][:,0][inds]
            y = self.adata.obsm['X_spatial'][:,1][inds]
            palette = self.color_palette[cluster]
            plt.scatter(x, y, s=s, marker='.', c = palette)
            plt.title(f'{color} cluster brain projection map with {self.adata.obs.shape[0]} cells')
            plt.axis("off")
        plt.show()

    def plot_brain_proj_per_cluster(self, color = 'leiden', s = 10):
        cluster_list = list(self.adata.obs[color].unique())
        for cluster in cluster_list:
            plt.figure(figsize = (10,8))
            plt.scatter(self.adata.obsm['X_spatial'][:,0], self.adata.obsm['X_spatial'][:,1], c = 'gray', s = 5, marker = '.')
            inds = self.adata.obs[color] == cluster
            x = self.adata.obsm['X_spatial'][:,0][inds]
            y = self.adata.obsm['X_spatial'][:,1][inds]
            palette = self.color_palette[cluster]
            plt.scatter(x, y, s=s, marker='.', c = palette)
            plt.title(f'{color} cluster: {cluster} brain projection map')
            plt.axis("off")
            plt.show()
        

    def plot_gene_umap(self, gene, size = 1, percentile = 95):
        plt.figure(figsize = (10,8))
        umap_x = self.adata.obsm['X_umap'][:,0]
        umap_y = self.adata.obsm['X_umap'][:,1]
        gene_idx = self.adata.var.index.get_loc(gene)
        gene_counts = self.adata.obsm['X_raw'][:, gene_idx]
        normalized_max = np.percentile(gene_counts, percentile)
        if normalized_max == 0.0:
            normalized_max = 1
        gene_counts[np.isnan(gene_counts)] = 0
        normalized_counts = np.clip(gene_counts / normalized_max, 0, 1)
        colors_mapped = cm.bwr(normalized_counts)
        plt.scatter(umap_x, umap_y, c = colors_mapped, s = size)
        plt.title(f'{gene} expression UMAP at {percentile}%')
        plt.axis('off')
        plt.show()

    def plot_gene_brain_proj(self, gene, size = 1, percentile = 95):
        plt.figure(figsize = (10,8))
        x = self.adata.obsm['X_spatial'][:,0]
        y = self.adata.obsm['X_spatial'][:,1]
        gene_idx = self.adata.var.index.get_loc(gene)
        gene_counts = self.adata.obsm['X_raw'][:, gene_idx]
        normalized_max = np.percentile(gene_counts, percentile)
        if normalized_max == 0.0:
            normalized_max = 1
        gene_counts[np.isnan(gene_counts)] = 0
        normalized_counts = np.clip(gene_counts / normalized_max, 0, 1)
        colors_mapped = cm.bwr(normalized_counts)
        plt.scatter(x, y, c = colors_mapped, s = size)
        plt.title(f'{gene} expression Brain Proj Map at {percentile}%')
        plt.axis('off')
        plt.show()

    def gene_expr_heatmap(self, gene_list, cluster = 'leiden', cmap = 'bwr', percentile = 95, plot_percentile = True):
        genes_of_interest = list(gene_list)
        gene_df = pd.DataFrame()
        for gene in [gene for gene in self.adata.var.index if gene in genes_of_interest]:
            idx = self.adata.var.index.get_loc(gene)
            gene_df[gene] = list(self.adata.obsm['X_raw'][:, idx])
        if plot_percentile == False:
            for gene in gene_df.columns:
                max = gene_df[gene].max()
                gene_df[gene] = gene_df[gene] / max
        else:
            for gene in gene_df.columns:
                max = np.percentile(gene_df[gene], percentile)
                gene_df[gene] = list(np.clip(gene_df[gene] / max, 0, 1))
        gene_df[cluster] = list(self.adata.obs[cluster])
        gene_df_means = gene_df.groupby(cluster).mean()
        sns.heatmap(gene_df_means, cmap = cmap)
        plt.title(f'Gene expression per {cluster} cluster')
        plt.show()
    
    def generate_color_dict(self, group = 'celltype', ref_dict = color_palette):
        celltype_list = list(self.adata.obs['celltype'].unique())
        celltype_color_list = ref_dict.values()
        celltype_dict = {}
        for celltype, color in zip(celltype_list, celltype_color_list):
            celltype_dict[celltype] = color
        celltype_dict = json.dumps(celltype_dict)
        print (celltype_dict)

    


