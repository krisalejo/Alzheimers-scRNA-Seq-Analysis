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

    


