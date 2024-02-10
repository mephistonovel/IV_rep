import numpy as np
import torch

class Allele():
    def __init__(self,model_id):
        self.model_id = model_id 
    
    def generate_zt(self,x_train, t_train):
        if self.model_id =='UAS':
            return x_train.mean(1).cpu().detach().numpy().astype(np.float16)
        elif self.model_id=='WAS':
            t_train = t_train.unsqueeze(1)
            weights = torch.corrcoef(torch.cat((x_train.T+1e-5,t_train.T),0))[-1,:-1]
            return (torch.sum(x_train*weights,axis=1)/torch.sum(weights)).cpu().detach().numpy().astype(np.float16)