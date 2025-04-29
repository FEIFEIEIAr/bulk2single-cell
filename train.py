import torch

import torch.autograd as autograd
import torch.nn.functional as F
import sc_bulk_model as models

from sklearn import preprocessing
import utils 
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn import svm, metrics
import os
# from captum.attr import IntegratedGradients, DeepLift, GradientShap
# from captum.attr import visualization as viz
import matplotlib.pyplot as plt

def diff_loss(z_p, z_s):
    s_l2_norm = torch.norm(z_s, p=2, dim=1, keepdim=True).detach()
    s_l2 = z_s.div(s_l2_norm.expand_as(z_s) + 1e-6)

    p_l2_norm = torch.norm(z_p, p=2, dim=1, keepdim=True).detach()
    p_l2 = z_p.div(p_l2_norm.expand_as(z_p) + 1e-6)
    diff_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))
    return diff_loss

def preTrain_AE(preTrain_model, n_epochs, train_dataloader, lr=1e-3, a=0.1):
    # device = utils.get_device()
    preTrain_model = preTrain_model
    optimizer = torch.optim.Adam(preTrain_model.parameters(), lr=lr)
    # warm_up_iter = 10
    # T_max = 50 #cycle
    # lr_max = 0.001 #max
    # lr_min = 1e-5 #min
    # warm-up + cosine anneal
    # lambda0 = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
    #     (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1
    early_stopping = utils.EarlyStopping(patience=5, delta=0.0001)
    # warm-up
    # lambda0 = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else cur_iter
    
    # # lambda1 = lambda cur_iter: 1
    
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)#, lambda1])
    # train_loss_l = []
    # test_loss_l = []
    # epoch_l = []
    for epoch in range(n_epochs):
        train_loss = 0
        preTrain_model.train()
        for n_batch, bulk_batch in enumerate(train_dataloader[0]):
            sc_batch = next(iter(train_dataloader[1]))
            x_sc = sc_batch
            x_bulk = bulk_batch
            #z_bulk_s, z_sc_s, z_bulk_p, z_sc_p, x_bulk_bar_1, x_sc_bar_1, x_bulk_bar_2, x_sc_bar_2
            em = preTrain_model(x_bulk, x_sc)
            diff_loss_2 = diff_loss(em[2], em[0]) + diff_loss(em[3], em[1])
            loss = F.mse_loss(em[4], x_bulk) + F.mse_loss(em[5], x_sc) + \
                    0.1 * F.mse_loss(em[6], x_bulk) + 0.1 * F.mse_loss(em[7], x_sc) + a * diff_loss_2
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(preTrain_model.parameters(), max_norm=3, norm_type=2)
            optimizer.step()
            # scheduler.step()
            train_loss += loss.item() / bulk_batch.shape[0]
        if epoch > 20:
            early_stopping(train_loss)
        if early_stopping.early_stop:
            print("Early stopping at epoch:",epoch)
            break
        # test_loss = eval_AE(preTrain_model, test_dataloader, device)
        # train_loss_l.append(train_loss)
        # test_loss_l.append(test_loss)
        # epoch_l.append(epoch)
        #if (epoch+1) % 50 == 0:
            #utils.draw_loss2(train_loss_l, test_loss_l, epoch_l)
    return preTrain_model    

# def eval_fcn(fcn_model, AE_model, labels, valid_dataloader,threshold):
#     device = utils.get_device()
#     fcn_model.to(device)
#     AE_model.to(device)
#     fcn_model.eval()
#     y_l = []
#     y_pred_l = []
#     y_pred_proba_l = []
#     for n_batch, (bt_data, index) in enumerate(valid_dataloader):
#         with torch.no_grad():
#             test_embedding  = AE_model.share_encode(bt_data.to(device))
#             # y_pred = fcn_model(bt_data).argmax(axis=1).type(dtype=torch.float32)
#             y_pred_proba = fcn_model(test_embedding).cpu().squeeze().numpy()
#             threshold = threshold
#             y_pred_labels = (y_pred_proba >= threshold).astype(int)
#             y = torch.tensor(labels[index.tolist()],dtype=torch.long)
#             y_l.extend(y.tolist())
#             y_pred_l.extend(y_pred_labels.tolist())
#             y_pred_proba_l.extend(y_pred_proba.tolist())
#             # correct = torch.sum(y_pred == y)
#     accuracy = metrics.accuracy_score(y_l, y_pred_l) 
#     F1 = metrics.f1_score(y_l, y_pred_l)
#     auc = metrics.roc_auc_score(y_l, y_pred_proba_l)
#     recall = metrics.recall_score(y_l, y_pred_l)
#     metrics_l = [accuracy,F1,auc,recall]
#     return metrics_l#correct.item() * 1.0 / len(y)

def eval_fcn(fcn_model, AE_model, labels, valid_dataloader):
    # device = utils.get_device()
    # fcn_model.to(device)
    # AE_model.to(device)
    fcn_model.eval()
    y_l = []
    y_pred_l = []
    for n_batch, (bt_data, index) in enumerate(valid_dataloader):
        with torch.no_grad():
            test_embedding  = AE_model.share_encode(bt_data)
            y_pred = fcn_model(test_embedding).argmax(axis=1).type(dtype=torch.float32).cpu().squeeze().numpy()
            y = torch.tensor(labels[index.tolist()],dtype=torch.float32)
            y_l.extend(y.tolist())
            y_pred_l.extend(y_pred.tolist())
            # correct = torch.sum(y_pred == y)
    accuracy = metrics.accuracy_score(y_l, y_pred_l) 
    F1 = metrics.f1_score(y_l, y_pred_l)
    return accuracy, F1

def train_fcn2(fcn_model, AE_model, n_epochs, y_train, 
               train_dataloader, test_dataloader, device, lr=1e-3):
    # fcn_model.to(device)
    # AE_model.to(device)
    optimizer = torch.optim.Adam(list(fcn_model.parameters()) + list(AE_model.parameters()), lr=lr)
    # warm_up_iter = 20
    # T_max = 50 #cycle
    # lr_max = 0.001 #max
    # lr_min = 1e-5 #min
    # lambda0 = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
    #     (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1
    # # lambda1 = lambda cur_iter: 1
    
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)#, lambda1])
    early_stopping = utils.EarlyStopping(patience=5, delta=0.0001)
    loss_fcn = torch.nn.CrossEntropyLoss()
    best_loss = 10000
    for epoch in range(n_epochs):
        fcn_model.train()
        AE_model.train()
        train_loss = 0 
        for step, (train_batch,index) in enumerate(train_dataloader):
            sc_batch = next(iter(test_dataloader))
            x_sc = sc_batch[0]
            x_bulk = train_batch
            #z_bulk_s, z_sc_s, z_bulk_p, z_sc_p, x_bulk_bar_1, x_sc_bar_1, x_bulk_bar_2, x_sc_bar_2
            em = AE_model(x_bulk, x_sc)
            diff_loss_2 = diff_loss(em[2], em[0]) + diff_loss(em[3], em[1])
            loss = F.mse_loss(em[4], x_bulk) + F.mse_loss(em[5], x_sc) + \
                    0.1 * F.mse_loss(em[6], x_bulk) + 0.1 * F.mse_loss(em[7], x_sc) + 0.1 * diff_loss_2
            train_embedding = em[0]
                
            y_pred = fcn_model(train_embedding)
            # y = torch.tensor(y_train[index.tolist()],dtype=torch.float32).unsqueeze(1).to(device)
            y = torch.tensor(y_train[index.tolist()],dtype=torch.long).to(device)
            loss2 = loss_fcn(y_pred, y)
            loss2.requires_grad_(True)
            loss_total = loss2 +0.1*loss
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(fcn_model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
            # scheduler.step()
            train_loss += loss2.item() / train_batch.shape[0]
        
        
        if early_stopping.early_stop:
            print("Early stopping at epoch:",epoch)
            break
        # metrics_l = eval_fcn(fcn_model, AE_model, y_val, val_dataloader)
        
        print('epoch:%d,train_loss: %.4f'%(epoch,loss2.item()))
        if epoch > 100:
            early_stopping(train_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            best_AE_model = AE_model
            best_fcn_model = fcn_model
        # acc_l.append(metrics_l[0])
        # auc_l.append(metrics_l[2])

        # if (epoch+1) % 50 == 0:
        #     print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | AUC {:.4f}".format(epoch+1, train_loss, metrics_l[0], metrics_l[2]))
    #utils.draw_single(acc_l,range(n_epochs),title='evaluation accuracy')
    #utils.draw_single(auc_l,range(n_epochs),title='evaluation AUC')
    #return metrics_l
    return best_fcn_model,best_AE_model

import anndata as ad
if __name__ == "__main__":
    random.seed(23333)
    device = utils.get_device()
    batch_size = 256
    n_epochs = 300
    fcn_n_epochs = 300
    n_hidden = [512,]
    fcn_hidden = [128, 32]
    critic_hidden = 32
    lr = 1e-3
    adv_train = False
    gdsc_label_threshold ='extreme'#'waterfall'#0.95#'median'##'median'#0.8
    a = 0.1
    dropout=0.4
    extreme_threshold = 0.30
    # n_top_genes = 3500
    # AE_model_dict ='./modelsave'
    # for i_pathways_threshold in [2,4,6,8,10,12,16,20,30]:
    pathways_threshold = 4  #i_pathways_threshold
    
    # Read bulk data
    bulk_path = './Data/filtered_gdsc.csv'
    pathways_path = "./Data/pathways/c5.go.cc.v2023.1.Hs.symbols.gmt"
    sc_data_folder = './Data/'
    
    
    datalist = ['GSE223779_fa34o_crizotinib_preprocessed','ts485to_crizotinib_preprocessed', 
                'PC9_gefitinib_preprocessed_2', 'SKMEL28_dabrafenib_preprocessed_2', 
                'GSE131984_jq1_preprocessed',
                'GSE156246_HCC1419_Lapatinib_preprocessed_genename',
                'GSE156246_BT474_Lapatinib_preprocessed_genename', 
                'GSE189324_Erlotinib_9weeks_preprocessed_genename',
                'GSE131984_paclitaxel_preprocessed', 
                'GSE108394_RAW', 'GSE163836_RAW_FCIBC02_paclitaxel',
                'GSE192575_RAW_human','GSE223003_RAW_Olaparib', 'GSE248717_RAW_MDAMB231_docetaxel',
                'GSE248717_RAW_SUM159_docetaxel','GSE248717_RAW_Vari068_docetaxel']
    # for n_top_genes in [1000,1500,2000,2500,3000]:
    
    filter_gdsc = pd.read_csv(bulk_path, index_col=0)    
    n_top_genes = 3500
    for data_idx in range(15):
        sc_drugdata = datalist[data_idx]
        drug_adata = ad.read('./Data/'+sc_drugdata+'.h5ad')
        bulk_adata = ad.AnnData(pd.read_csv('./Data/filtered_gdsc_expr.csv',index_col=0))
        bulk_adata.obs['CellLineName'] = bulk_adata.obs.index.values
        sc_adata = ad.read('./Data/filter_pan_cancer.h5ad')
        
        drug_adata.var_names_make_unique()
        same_gene = utils.filter_gene(bulk_adata, sc_adata, drug_adata)
        bulk_adata = bulk_adata[:,same_gene]
        sc_adata = sc_adata[:,same_gene]
        drug_adata = drug_adata[:,same_gene]
        
        hvg = utils.choice_gene(bulk_adata, sc_adata, n_top_genes=n_top_genes)
        
        bulk_X = bulk_adata[:,hvg].X
        sc_X = sc_adata[:,hvg].X
    
        pathway_dict = utils.read_gmt(pathways_path, min_g=0, max_g=1000)
        
        # pathway_dict = OrderedDict(np.load('cellmarker.npy', allow_pickle=True).item())
        pathway_mask = utils.create_pathway_mask(hvg, pathway_dict, add_missing=5, fully_connected=True)
        non_zero_counts = np.sum(pathway_mask != 0, axis=0)
        valid_columns = np.where(non_zero_counts > pathways_threshold)[0]
        new_array = pathway_mask[:, valid_columns]
        
        #get feature names
        pathway_names = []
        for k,v in pathway_dict.items():
            pathway_names.append(k)
        feature_names = [pathway_names[i] for i in valid_columns[:-5]]
        final_feature_names = np.array(feature_names + ["fcn1","fcn2","fcn3","fcn4","fcn5"])
        jq1_list = ['GSE131135_SUM159R_jq1_preprocessed_genename', 'GSE131135_SUM159_jq1_preprocessed_genename',
                    'GSE131135_SUM149_jq1_preprocessed_genename', 'GSE131135_SUM149R_jq1_preprocessed_genename',
                    'GSE131984_jq1_preprocessed']
        
        if sc_drugdata in jq1_list: 
            drug_name = 'JQ1'
        elif sc_drugdata in ['ts485to_crizotinib_preprocessed','GSE223779_fa34o_crizotinib_preprocessed']: 
            drug_name = 'Crizotinib'
        elif sc_drugdata in ['GSE131984_paclitaxel_preprocessed', 'GSE163836_RAW_FCIBC02_paclitaxel']: 
            drug_name = 'Paclitaxel'
        elif sc_drugdata in ['GSE156246_BT474_Lapatinib_preprocessed_genename', 
                             'GSE156246_HCC1419_Lapatinib_preprocessed_genename']: 
            drug_name = 'Lapatinib'
        elif sc_drugdata == 'SKMEL28_dabrafenib_preprocessed_2': 
            drug_name = 'Dabrafenib'
        elif sc_drugdata == 'PC9_gefitinib_preprocessed_2': 
            drug_name = 'Gefitinib'
        elif sc_drugdata == 'GSE108394_RAW':
            drug_name = 'PLX-4720'
        elif sc_drugdata == 'GSE223003_RAW_Olaparib':
            drug_name = 'Olaparib'
        elif sc_drugdata == 'GSE192575_RAW_human':
            drug_name = 'Cisplatin'
        else:
            raise Exception("Couldn't found the drug ID")
        
        test_labels = drug_adata.obs['sensitive'].tolist()
        drug_adata = drug_adata[:,hvg]
        try:
            test_X = drug_adata.X.toarray()
        except:
            test_X = drug_adata.X
        gdsc_info = filter_gdsc.loc[drug_name]
        train_labels = []
        if type(gdsc_label_threshold) is float:
            for i in range(gdsc_info.shape[0]):#get binary labels based on a threshold of AUC values.
                if gdsc_info['AUC'][i] > gdsc_label_threshold:
                    train_labels.append(0)
                else:
                    train_labels.append(1)
        elif gdsc_label_threshold == 'median':
            median = gdsc_info['AUC'].median()
            for i in range(gdsc_info.shape[0]):#get binary labels based on a threshold of AUC values.
                if gdsc_info['AUC'][i] > median:
                    train_labels.append(0)
                else:
                    train_labels.append(1)
        elif gdsc_label_threshold == 'waterfall':   
            #median is adopted when the IC50 values are linearly dependent, 
            #else the argmax dis (np.square(neg_log_x-min_) + np.square(neg_log_x-max_))
            #1) for linear curves (whose regression line fitting has a Pearson correlation
            # >0.95), the sensitive/resistant cutoff of AUC values is the median
            # among all cell lines; 2) otherwise, the cutoff is the AUC value of a
            # specific boundary data point, which has the largest distance to a line
            # linking two datapoints having the largest and smallest AUC values.
            threshold = utils.callWaterfall(gdsc_info['AUC'], min_linear=0.95, dis='2line') 
            for i in range(gdsc_info.shape[0]):#get binary labels based on a threshold of AUC values.
                if gdsc_info['AUC'][i] > threshold:
                    train_labels.append(0)
                else:
                    train_labels.append(1)
                    
        elif gdsc_label_threshold == 'extreme':
            unfilter_data = gdsc_info['AUC']
            low_indices, high_indices = utils.get_extreme_indices(unfilter_data, threshold=extreme_threshold)
            resis_sample = gdsc_info.iloc[high_indices,:]
            sensi_sample = gdsc_info.iloc[low_indices,:]
            new_gdsc_info = pd.concat([sensi_sample,resis_sample])
            
            lower_threshold = unfilter_data.quantile(extreme_threshold)
            lower_threshold_2 = unfilter_data.quantile(extreme_threshold+0.05)
            upper_threshold = unfilter_data.quantile(1-extreme_threshold)
            upper_threshold_2 = unfilter_data.quantile(0.95-extreme_threshold)
            
            low_indices_val = [i for i, x in enumerate(unfilter_data) if x < lower_threshold_2 and x > lower_threshold]
            high_indices_val = [i for i, x in enumerate(unfilter_data) if x < upper_threshold and x >upper_threshold_2]
            val_resis_sample = gdsc_info.iloc[high_indices_val,:]
            val_sensi_sample = gdsc_info.iloc[low_indices_val,:]
            val_new_gdsc_info = pd.concat([val_sensi_sample,val_resis_sample])
        test_labels = np.array(test_labels)  
        
        if gdsc_label_threshold == 'extreme':
            y_train = np.concatenate([np.ones(sensi_sample.shape[0]),np.zeros(resis_sample.shape[0])])
            train_X = pd.DataFrame(bulk_X, index = bulk_adata.obs['CellLineName'])
            X_train = np.array(train_X.loc[new_gdsc_info['Cell Line Name'].tolist()])
            
            # y_val = np.concatenate([np.ones(val_sensi_sample.shape[0]),np.zeros(val_resis_sample.shape[0])])
            # val_X  = pd.DataFrame(bulk_X, index = bulk_adata.obs['CellLineName'])
            # X_val = np.array(val_X.loc[val_new_gdsc_info['Cell Line Name'].tolist()])
        else:
            train_labels = np.array(train_labels)    
              
            train_X = pd.DataFrame(bulk_X, index = bulk_adata.obs['CellLineName'])
            train_X = np.array(train_X.loc[gdsc_info['Cell Line Name'].tolist()])
            
    # if random_sampling is True:
        # random_number = random.sample(range(sc_X.shape[0]),int(0.1*sc_X.shape[0]))
        # sc_X = sc_X[random_number,:]
        # accuracy_l=[]
        # auc_l = []
        # recall_l=[]
        # f1_l = []
        acc_l = []
        f1_l = []
        for j in range(5):
            random.seed(j)
            # sc_train_X, sc_test_X = train_test_split(sc_X, test_size=0.1)
            
            bulk_train_dataloader = DataLoader(torch.tensor(bulk_X, dtype = torch.float32).to(device), 
                                                       batch_size=batch_size, shuffle=True)
            #bulk_test_dataloader = DataLoader(torch.tensor(bulk_test_X, dtype = torch.float32), 
            #                                  batch_size=batch_size, shuffle=True)
            sc_train_dataloader = DataLoader(torch.tensor(preprocessing.scale(sc_X), dtype = torch.float32).to(device), 
                                             batch_size=batch_size, shuffle=True)
            #sc_test_dataloader = DataLoader(torch.tensor(preprocessing.scale(sc_test_X), dtype = torch.float32), 
            #                                batch_size=batch_size, shuffle=True)
            
            train_dataloader = (bulk_train_dataloader, sc_train_dataloader)
            # test_dataloader = (bulk_test_dataloader, sc_test_dataloader)
            
            # preTrain_model = models.model3(new_array, n_hidden = n_hidden, dropout=dropout).to(device)
            AE_model = models.model3(new_array, n_hidden = n_hidden, dropout=dropout).to(device)
            # pretrain_dict = AE_model_dict+'/GO_%d_AE_model.pkl'%(pathways_threshold)
            # if os.path.exists(pretrain_dict):
            #     try:
            #         print('Load model')
            #         model_state_dict = torch.load(pretrain_dict)
            #         preTrain_model.load_state_dict(model_state_dict)
            #     except:
            #         print('Load model unsuccessfully')
            # else:
            print('Start pretraining')
            # preTrain_model = preTrain_AE(preTrain_model, n_epochs, train_dataloader, lr, a=a)
                # torch.save(preTrain_model.state_dict(), pretrain_dict)
        
            # AE_model = preTrain_model
        
            #drug response classification module
            # X_train, X_val, y_train, y_val = train_test_split(train_X, train_labels, test_size=0.15)
            train_data = TensorDataset(torch.tensor(X_train,dtype=torch.float32).to(device),
                                        torch.tensor(range(X_train.shape[0])))
            test_data = TensorDataset(torch.tensor(test_X,dtype=torch.float32).to(device),
                                        torch.tensor(range(test_X.shape[0])))
            
            train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            # val_data = torch.tensor(X_val,dtype=torch.float32).to(device)
            test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)    
            
            fcn_model = models.fcn2(new_array.shape[1], fcn_hidden, dropout=dropout).to(device)
            
            best_fcn_model,best_AE_model = train_fcn2(fcn_model, AE_model, fcn_n_epochs, y_train, 
                                                       train_dataloader, test_dataloader, device,lr=1e-3)
            
            # best_fcn_model.eval()
            # with torch.no_grad():
                # test_embedding  = best_AE_model.share_encode(val_data.to(device))
                # y_pred = fcn_model(bt_data).argmax(axis=1).type(dtype=torch.float32)
                # y_pred_proba = best_fcn_model(test_embedding).cpu().squeeze().numpy()
            # fpr_dl, tpr_dl, thresholds_dl = metrics.roc_curve(y_val, y_pred_proba)
            # j_scores = tpr_dl - fpr_dl
            
            # Find threshold with maximum J-score
            # best_threshold = thresholds_dl[np.argmax(j_scores)]
            # print(f'Best threshold: {best_threshold}')
                # y_pred_proba_dl = model(X_val_tensor).squeeze().numpy()
            accuracy, F1 = eval_fcn(best_fcn_model, best_AE_model, test_labels, test_dataloader)
            print("Filnally metrics: dataset: {} | Accuracy {:.4f} | F1 {:.4f}".format(sc_drugdata, accuracy, F1))
            acc_l.append(accuracy)
            f1_l.append(F1)
        if data_idx == 0:
            df_acc = pd.DataFrame(acc_l,index=range(5),columns=[sc_drugdata,])
            df_f1 = pd.DataFrame(f1_l,index=range(5),columns=[sc_drugdata,])
        else:
            df_acc[sc_drugdata] = acc_l
            df_f1[sc_drugdata] = f1_l
        df_acc.to_csv('./results/extreme0.2_top3500_wopretrain_acc.csv')
        df_f1.to_csv('./results/extreme0.2_top3500_wopretrain_f1.csv')
                