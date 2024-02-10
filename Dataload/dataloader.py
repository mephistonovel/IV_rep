import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def data_load_syn_low(args, n_samples, batch_size, cuda, device):
    treatment = args.treatment
    response = args.response
    nodep = args.dependency
    nointer = args.interaction
    #args.baseline
    assert args.baseline == 0
    assert args.highdim == 0

    data = pd.read_csv(f'./Dataload/Data/Syn_{n_samples}_{treatment}_{response}/Syn_train_{nodep}_{nointer}_{args.true_effect}.csv')
    
        
    t, y = data.loc[:, 'X'].values, data.loc[:, 'Y'].values # W, Y
    x = data.filter(regex='^D').values
    

    if args.baseline == 1 and treatment == 'b':
        ym, ys = y.mean(), y.std()
        y = (y-ym)/ys


    x = torch.from_numpy(x)
    t = torch.from_numpy(t).squeeze()
    y = torch.from_numpy(y).squeeze()



    #cuda로 보내기
    if cuda:
        x = x.to(device)
        y = y.to(device)
        t = t.to(device)
    train = (x, t, y)
    data_loader_train = DataLoader(TensorDataset(x,t,y), batch_size=batch_size, 
                          shuffle=True, num_workers=0)
    
    
    ### test set ### 
    data_test = pd.read_csv(f'./Dataload/Data/Syn_{n_samples}_{treatment}_{response}/Syn_test_{nodep}_{nointer}_{args.true_effect}.csv')
     
    t_test, y_test = data_test.loc[:, 'X'].values, data_test.loc[:, 'Y'].values # W, Y
    x_test = data_test.filter(regex='^D').values
    d1 = data_test.filter(regex='^D1').values
    x_test = torch.from_numpy(x_test)
    t_test = torch.from_numpy(t_test).squeeze()
    y_test = torch.from_numpy(y_test).squeeze()
    
    if args.baseline == 1 and treatment == 'b':
        ym_test, ys_test = y_test.mean(), y_test.std()
        y_test = (y_test-ym_test)/ys_test
        
    if cuda:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        t_test = t_test.to(device)

    test = (x_test, t_test, y_test)
    data_loader_test = DataLoader(TensorDataset(x_test,t_test,y_test), batch_size=batch_size, 
                          shuffle=True, num_workers=0)
    
    return train,data_loader_train, test,data_loader_test, d1

def data_load_syn_high(args, n_samples, batch_size, cuda, device):
    treatment = args.treatment
    response = args.response
    nodep = args.dependency
    nointer = args.interaction
    #args.baseline
    assert args.baseline == 0
    assert args.highdim == 1
    ### Train set ### 

    data = pd.read_csv(f'./Dataload/Data/Syn_highdim_{n_samples}_{treatment}_{response}/Syn_train_{nodep}_{nointer}_{args.true_effect}.csv')

    t, y = data.loc[:, 'X'].values, data.loc[:, 'Y'].values # W, Y
    x = data.filter(regex='^D').values
    

    if args.baseline == 1 and treatment == 'b':
        ym, ys = y.mean(), y.std()
        y = (y-ym)/ys


    x = torch.from_numpy(x)
    t = torch.from_numpy(t).squeeze()
    y = torch.from_numpy(y).squeeze()



    #cuda로 보내기
    if cuda:
        x = x.to(device)
        y = y.to(device)
        t = t.to(device)
    train = (x, t, y)
    data_loader_train = DataLoader(TensorDataset(x,t,y), batch_size=batch_size, 
                          shuffle=True, num_workers=0)
    
    
    ### test set ### 
    data_test = pd.read_csv(f'./Dataload/Data/Syn_highdim_{n_samples}_{treatment}_{response}/Syn_test_highdim_d2dim100_dimc50_{args.true_effect}.csv')
   
    t_test, y_test = data_test.loc[:, 'X'].values, data_test.loc[:, 'Y'].values # W, Y
    x_test = data_test.filter(regex='^D').values
    # d1 = data_test.filter(regex='^D1').values
    d1=0 
    x_test = torch.from_numpy(x_test)
    t_test = torch.from_numpy(t_test).squeeze()
    y_test = torch.from_numpy(y_test).squeeze()
    
    if args.baseline == 1 and treatment == 'b':
        ym_test, ys_test = y_test.mean(), y_test.std()
        y_test = (y_test-ym_test)/ys_test
        
    if cuda:
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        t_test = t_test.to(device)

    test = (x_test, t_test, y_test)
    data_loader_test = DataLoader(TensorDataset(x_test,t_test,y_test), batch_size=batch_size, 
                          shuffle=True, num_workers=0)
    
    return train,data_loader_train, test,data_loader_test, d1


def data_load_real(args, batch_size, cuda, device, target='black'):
    if args.treatment == 'con':
        full_data = pd.io.stata.read_stata('./Dataload/main_data_all.dta')
        full_data = full_data.dropna(subset=['styr', 'P2', 'xid', 'Male_Pct', 'Female_HH_Pct', 'Poverty', 'Unemployment_Rate', 'Median_HH_Income', 'NH_White_Pct', 'NH_Black_Pct', 'Hispanic_Pct', 'Age_0_14_Pct', 'Age_15_24_Pct', 'Age_25_44_Pct', 'Age_45_Plus_Pct', 'Z',f'llarrest_tot_{target}', 'S'])
        data=full_data.groupby('xid').sample(frac=0.8,random_state=0) #random state is a seed value
        data_test=full_data.drop(data.index)
        if target == 'black':
            t, y = data.loc[:, 'S'].values, data.loc[:, 'llarrest_tot_black'].values # W, Y
        elif target == 'white':
            t, y = data.loc[:, 'S'].values, data.loc[:, 'llarrest_tot_white'].values # W, Y
        x = data.loc[:,['styr', 'P2', 'xid', 'Male_Pct', 'Female_HH_Pct', 'Poverty', 'Unemployment_Rate', 'Median_HH_Income', 'NH_White_Pct', 'NH_Black_Pct', 'Hispanic_Pct', 'Age_0_14_Pct', 'Age_15_24_Pct', 'Age_25_44_Pct', 'Age_45_Plus_Pct','Z']]
        
        # x = data.loc[:,['styr','Male_Pct', 'Female_HH_Pct', 'Poverty', 'Unemployment_Rate', 'Median_HH_Income', 'NH_White_Pct', 'NH_Black_Pct', 'Hispanic_Pct', 'Age_0_14_Pct', 'Age_15_24_Pct', 'Age_25_44_Pct', 'Age_45_Plus_Pct']]
        x = pd.get_dummies(x, columns=['xid'],dtype=int).values
        # x = x.values
        x = torch.from_numpy(x)
        t = torch.from_numpy(t).squeeze()
        y = torch.from_numpy(y).squeeze()
        
        
 
        #cuda로 보내기
        if cuda:
            x = x.to(device)
            y = y.to(device)
            t = t.to(device)
        train = (x, t, y)
        data_loader_train = DataLoader(TensorDataset(x,t,y), batch_size=batch_size,
                            shuffle=True, num_workers=0)
        ## test set ###
        if target == 'black':
            t_test, y_test = data_test.loc[:, 'S'].values, data_test.loc[:, 'llarrest_tot_black'].values # W, Y
        elif target == 'white':
            t_test, y_test = data_test.loc[:, 'S'].values, data_test.loc[:, 'llarrest_tot_white'].values # W, Y
        # d1 = data_test.filter(regex='^D1').values
        d1=0
        x_test = data_test.loc[:,['styr', 'P2', 'xid', 'Male_Pct', 'Female_HH_Pct', 'Poverty', 'Unemployment_Rate', 'Median_HH_Income', 'NH_White_Pct', 'NH_Black_Pct', 'Hispanic_Pct', 'Age_0_14_Pct', 'Age_15_24_Pct', 'Age_25_44_Pct', 'Age_45_Plus_Pct','Z']]
        # x_test = data_test.loc[:,['styr', 'Male_Pct', 'Female_HH_Pct', 'Poverty', 'Unemployment_Rate', 'Median_HH_Income', 'NH_White_Pct', 'NH_Black_Pct', 'Hispanic_Pct', 'Age_0_14_Pct', 'Age_15_24_Pct', 'Age_25_44_Pct', 'Age_45_Plus_Pct']].values
        
        x_test = pd.get_dummies(x_test, columns=['xid'],dtype=int).values
        x_test = torch.from_numpy(x_test)
        t_test = torch.from_numpy(t_test).squeeze()
        y_test = torch.from_numpy(y_test).squeeze()
        
        
        
        if cuda:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            t_test = t_test.to(device)
        test = (x_test, t_test, y_test)
        data_loader_test = DataLoader(TensorDataset(x_test,t_test,y_test), batch_size=batch_size,
                            shuffle=True, num_workers=0)
        return train,data_loader_train, test,data_loader_test, d1
    else:
        data = pd.io.stata.read_stata('./Dataload/401ksubs.dta')
        
        data_train, data_test = train_test_split(data, test_size=0.2, stratify=data['e401k'], random_state=0)

        # data_train=data.sample(frac=0.8,random_state=0) #random state is a seed value
        # data_test=data.drop(data_train.index)
        t, y = data_train.loc[:, 'e401k'].values, data_train.loc[:, 'pira'].values # W, Y
        x = data_train.drop(columns = ['e401k', 'pira','marr','male','p401k']).values
        x = torch.from_numpy(x)
        t = torch.from_numpy(t).squeeze()
        y = torch.from_numpy(y).squeeze()
        
        xm,xs = x.mean(0),x.std(0)
        x = (x-xm)/xs 
        # tm, ts = t.mean(),t.std()
        # t = (t-tm)/ts
        # ym, ys = y.mean(), y.std()
        # y = (y-ym)/ys
        if cuda:
                    x = x.to(device)
                    y = y.to(device)
                    t = t.to(device)
        train = (x, t, y)
        data_loader_train = DataLoader(TensorDataset(x,t,y), batch_size=batch_size,
                            shuffle=True, num_workers=0)
        ## test set ###
        t_test, y_test = data_test.loc[:, 'e401k'].values, data_test.loc[:, 'pira'].values # W, Y
        # d1 = data_test.filter(regex='^D1').values
        d1=0
        x_test = data_test.drop(columns = ['e401k', 'pira','marr','male','p401k']).values
        x_test = torch.from_numpy(x_test)
        t_test = torch.from_numpy(t_test).squeeze()
        y_test = torch.from_numpy(y_test).squeeze()
        
        xm_test,xs_test = x_test.mean(0),x_test.std(0)
        x_test= (x_test-xm_test)/xs_test 
        if cuda:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            t_test = t_test.to(device)
        test = (x_test, t_test, y_test)
        data_loader_test = DataLoader(TensorDataset(x_test,t_test,y_test), batch_size=batch_size,
                            shuffle=True, num_workers=0)
        return train,data_loader_train, test,data_loader_test, d1
