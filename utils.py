from collections import OrderedDict
import numpy as np
import pandas as pd
import scanpy as sc
import math
import torch

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def get_hvg_list(adata):
    #get highly variable genes from a calculated adata.
    l = []
    for i in range(adata.shape[1]):
        if adata.var.highly_variable[i]:
            l.append(adata.var.highly_variable.index[i])
    return l

def filter_gene(bulk_adata, sc_adata, test_adata):
    gene_1 = bulk_adata.var.index.tolist()
    gene_2 = sc_adata.var.index.tolist()
    gene_3 = test_adata.var.index.tolist()
    same_gene = list(set(gene_1)&set(gene_2)&set(gene_3))
    return same_gene

def choice_gene(bulk_adata, sc_adata, n_top_genes=2000):#, test_adata):
    sc.pp.normalize_per_cell(sc_adata)
    sc.pp.log1p(sc_adata)
    sc.pp.normalize_per_cell(bulk_adata)
    sc.pp.log1p(bulk_adata)
    sc.pp.highly_variable_genes(bulk_adata, n_top_genes=n_top_genes)
    sc.pp.highly_variable_genes(sc_adata, n_top_genes=n_top_genes)
    # sc.pp.highly_variable_genes(test_adata, n_top_genes=2000)
    hvg = set(get_hvg_list(bulk_adata)) | set(get_hvg_list(sc_adata))
    return list(hvg)

def get_device(use_cpu=False):
    """[summary]
    Returns:
        [type]: [description]
    """
    if torch.cuda.is_available() and use_cpu == False:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def preprocessing_h5ad(adata, nb_genes=None):
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    if nb_genes is not None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4,
                                    min_disp=0.5, n_top_genes=nb_genes, subset=True)
    return adata
def use_Leiden(features, labels=None, n_neighbors=20, resolution=1):
    #from https://github.com/eleozzr/desc/blob/master/desc/models/network.py line 241
    adata0=sc.AnnData(features)
    sc.pp.pca(adata0)
    if n_neighbors==0:
        sc.pp.neighbors(adata0, knn=False, method = 'gauss', metric='cosine', n_pcs=0)
    else:
        sc.pp.neighbors(adata0, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.leiden(adata0, resolution=resolution)
    sc.tl.umap(adata0)
    sc.pl.umap(adata0,color=['leiden'])
    if labels is not None:
        asw = np.round(silhouette_score(features, adata0.obs['leiden']), 3)
        nmi = np.round(normalized_mutual_info_score(labels, adata0.obs['leiden']), 3)
        ari = np.round(adjusted_rand_score(labels, adata0.obs['leiden']), 3)
        return asw, ari, nmi

def sort_encode_data(embedding, _ID):
    dat = pd.DataFrame(embedding.cpu().detach().numpy())
    _ID = pd.DataFrame(_ID.cpu().detach().numpy(),columns=['ID',])
    dat_ID = pd.concat([dat,_ID],axis=1)
    sort_dat = np.asarray(dat_ID.sort_values(by='ID',ascending=True).drop('ID',axis=1).values)
    return sort_dat

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of pathway:genes.
    min_g and max_g are optional gene set size filters.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=True, fully_connected=True, to_tensor=False):
    """ Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.
    Expects a list of genes and pathway dict.
    If add_missing is True or an int, input genes that are not part of pathways are all connected to a "placeholder" pathway.
    If fully_connected is True, all input genes are connected to the placeholder units.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted."""
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    for j, k in enumerate(dict_pathway.keys()):
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask

def draw_single(y,x,title='train/test loss'):
    # plt.cla()
    x = np.array(x)
    y = np.array(y)
    plt.title(title)
    line1, = plt.plot(x,y,color='steelblue')
    plt.xlabel('epoches')
    plt.ylabel('y')
    plt.show()
    
def draw_loss2(Loss_list1,Loss_list2,epoch,title='train/test loss', legend=['train_loss', 'test_loss']):
    # plt.cla()
    x1 = np.array(epoch)
    y1 = np.array(Loss_list1)
    y2 = np.array(Loss_list2)
    plt.title(title)
    line1, = plt.plot(x1,y1,color='brown')
    line2, = plt.plot(x1,y2,color='steelblue')
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.legend(handles = [line1, line2], labels = legend)
    plt.show()
    
# import numpy as np
from scipy.stats import pearsonr
# import pandas as pd
# import matplotlib.pyplot as plt

def callWaterfall(x, 
                  index_type="AUC",
                  min_linear = 0.95,
                  plot=True,
                  dis='2max&min'):
    
    x = x[~np.isnan(x)]
    # sort log_IC50
    # if  all(i>0 for i in x) > 0:
    #     neg_log_x = -1 * np.log10(x)
    # else:
    #     neg_log_x = -1 *   x#-1 * np.log10(x)
    x_sort = x[np.argsort(x)[::-1]]
    min_ = np.min(x)
    max_ = np.max(x)
    
    linear_fit_x = [i for i in np.linspace(max_, min_, num=len(x_sort), endpoint=True)]
    pcc = pearsonr(x_sort, linear_fit_x)[0]
    if pcc >= min_linear:
        cutoff = np.median(x)
    elif dis == '2max&min':
        dis = np.sqrt(np.square(x-min_) + np.square(x-max_))
        ind = np.argmin(x) if np.argmax(dis) == np.argmax(x) else np.argmax(dis)
        cutoff = x[ind]
    elif dis == '2line':
        # a = np.array([0,max_])
        # b = np.array([1,min_])
        dis = []
        theta = math.atan(max_-min_)#/int(len(x)))
        for i in range(len(x)):
            dis.append(np.array((x_sort[i]-linear_fit_x[i])*math.cos(theta)))
            # p = np.array([i/int(len(x)),x[i]])
            # dis.append(dis_point_to_seg_line(p, a, b))
            
        ind = np.argmax(np.array(dis))
        cutoff = x_sort[ind]
        
    if plot:
        plotWaterfall(np.array([i for i in range(1, len(x)+1)]), 
                      index_type,
                      x_sort, 
                      cutoff=cutoff, 
                      pcc=pcc)
    return cutoff
    
# def dis_point_to_seg_line(p,a,b):
#     d = np.divide(b-a, np.linalg.norm(b-a))
#     s=np.dot(a-p,d)
#     t=np.dot(p-b,d)
#     h=np.maximum.reduce([s,t,0])
#     c=np.cross(p-a,d)
#     return np.hypot(h,np.linalg.norm(c))

def plotWaterfall(index, 
                  index_type,
                  y,
                  cutoff,
                  pcc):
    
    plt.figure(figsize=(7,4), dpi=300)
    plt.scatter(index[y>cutoff], y[y>cutoff], c="#06d6a0", label="resistant(n=%d)" % np.sum(y>cutoff))
    plt.scatter(index[y<=cutoff], y[y<=cutoff], c="#ffd166", label="sensitive(n=%d)" % np.sum(y<=cutoff))
    plt.scatter(index[y==cutoff], y[y==cutoff], c="#ef476f", label="cutoff")
    plt.plot(index, 
             [i for i in np.linspace(np.max(y), np.min(y), num=len(y), endpoint=True)],
             "gray")
    plt.xlabel("index")
    plt.ylabel("%s"%(index_type))
    plt.legend(loc="upper right")
    plt.text(0, min(y)+0.1, "R=%.4f"%pcc)
    plt.title("Waterfall Distribution")
    plt.show()
    
    
def plot_violin(y,y_pred):
    sensitive_cell = []
    resistant_cell = []
    for i in y:
        if y == 0:
            resistant_cell.append(y_pred[i])
        elif y == 1:
            sensitive_cell.append(y_pred[i])
    plt.subplot(1,2,1)  
    plt.violinplot(np.array(sensitive_cell))      
    plt.title('sensitive_cell')
    plt.subplot(1,2,2)  
    plt.violinplot(np.array(resistant_cell))      
    plt.title('resistant_cell')
    plt.show()
    
def get_extreme_indices(data, threshold):
    """
    获取数据中处于较低和较高阈值范围内的位置
    """
    lower_threshold = data.quantile(threshold)
    upper_threshold = data.quantile(1-threshold)
    low_indices = [i for i, x in enumerate(data) if x < lower_threshold]
    high_indices = [i for i, x in enumerate(data) if x >= upper_threshold]
    return low_indices, high_indices

# 定义早停类
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def umap_visual(embedding, title=None, save_path=None, label=None):
    n_lables = len(set(label)) + 1
    # ARI = calcu_ARI(label, true_label)
    # NMI = normalized_mutual_info_score(true_label, label)
    xlim_l = int(embedding[:, 0].min()) - 2
    xlim_r = int(embedding[:, 0].max()) + 2
    ylim_d = int(embedding[:, 1].min()) - 2
    ylim_u = int(embedding[:, 1].max()) + 2
    plt.figure(figsize = (6,4), dpi=300)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=3)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar()
    plt.xlim((xlim_l, xlim_r))
    plt.ylim((ylim_d, ylim_u))
    plt.title(title)
    plt.grid(False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()