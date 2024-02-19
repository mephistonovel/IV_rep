from .Polytsls.polytsls import poly_tsls
from .Polytsls.polytsls_v2 import poly_tsls_est
from .Onesiv.onesiv import osiv
from .KernelIV.kerneliv import kiv 
from .KernelIV.kerneliv_v2 import kiv_est

from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from econml.iv.dml import OrthoIV
from econml.iv.dml import DMLIV
from econml.iv.dr import LinearDRIV,DRIV
from linearmodels.iv import IV2SLS,IVGMM
import pandas as pd
import numpy as np
from sklearn.svm import SVC

Est = {
    'tsls': IV2SLS,
    'IVGMM': IVGMM,
    'DML': DMLIV,
    'Ortho': OrthoIV,
    'Poly_tsls': poly_tsls,
    'KernelIV': kiv
}

Est2 = {
    'tsls': IV2SLS,
    'IVGMM': IVGMM,
    'DML': DMLIV,
    'Ortho': OrthoIV,
    'Poly_tsls': poly_tsls_est,
    'KernelIV': kiv_est
}

def estimate_report(treatment, response, zt_train, t_train, y_train, 
             zt_test, t_test, y_test, x_condition_train=None, x_condition_test=None):
    estimates = {}
    for est in Est2.keys():
        est_train = ce_estimator_report(est_id = est,treatment_type = treatment,response=response)
        est_train.fit(Y=y_train, T=t_train, Z = zt_train,C=x_condition_train)
        effect_train = est_train.effect_estimate()
        
        est_test = ce_estimator_report(est_id = est,treatment_type = treatment,response = response)
        est_test.fit(Y=y_test, T=t_test, Z = zt_test,C=x_condition_test)
        effect_test = est_test.effect_estimate()
        
        train_key = est+'_train_MAE'
        test_key = est+'_test_MAE'


        estimates[train_key] = effect_train
        estimates[test_key] = effect_test

        
    return pd.DataFrame(estimates,columns = estimates.keys(),index=[0])     

def estimate_abs(treatment, response, true_effect, zt_train, t_train, y_train, 
             zt_test, t_test, y_test, x_condition_train=None, x_condition_test=None):
    estimates = {}
    for est in Est.keys():
        est_train = ce_estimator(est_id = est,treatment_type = treatment,response=response)
        est_train.fit(Y=y_train, T=t_train, Z = zt_train,C=x_condition_train)
        effect_train = est_train.effect_estimate()
        
        est_test = ce_estimator(est_id = est,treatment_type = treatment,response = response)
        est_test.fit(Y=y_test, T=t_test, Z = zt_test,C=x_condition_test)
        effect_test = est_test.effect_estimate()
        
        train_key = est+'_train_MAE'
        test_key = est+'_test_MAE'
        if response == 'linear':
            if est in ['DML','Ortho','LinearDRIV']:
                y0hat_train = true_effect
                y0hat_test = true_effect
            else:
                y0hat_train = true_effect*t_train
                y0hat_test = true_effect*t_test
        else:
            effect_train = effect_train
            effect_test = effect_test
            if est in ['DML','Ortho','LinearDRIV','IVGMM','tsls']:
                y0hat_train = np.exp(0.5)-1
                y0hat_test = np.exp(0.5)-1
            else:
                y0hat_train = np.exp(0.5*t_train)
                y0hat_test = np.exp(0.5*t_test)
                
        estimates[train_key] = np.mean(np.abs(effect_train-y0hat_train))
        estimates[test_key] = np.mean(np.abs(effect_test-y0hat_test))
        
    return pd.DataFrame(estimates,columns = estimates.keys(),index=[0])     
    

class ce_estimator:
    def __init__(self,est_id,treatment_type,response):
        self.est_id = est_id
        self.est = Est[est_id]
        self.treatment_type = treatment_type
        self.response = response
        if est_id == ['tsls', 'IVGMM','Poly_tsls','KernelIV']:
            pass
        elif est_id in ['LinearDRIV']:
            self.est_model = self.est(discrete_treatment=(self.treatment_type=='b'),discrete_instrument=False)
        elif est_id in ['DML','DRIV']:
            self.est_model = self.est(discrete_treatment=(self.treatment_type=='b'),discrete_instrument=False)
        elif est_id == 'Ortho':
            self.est_model = self.est(projection=False, discrete_treatment=(self.treatment_type=='b'), discrete_instrument=False)
        elif est_id == 'NPDML':
            self.est_model = self.est(discrete_treatment=(self.treatment_type=='b'),discrete_instrument=False,model_final=StatsModelsLinearRegression())

    def fit(self,Z,T,Y,C=None):
        
        if self.est_id == 'tsls' or self.est_id == 'IVGMM':
            self.est_model = self.est(dependent = Y, exog = C,endog=T,instruments=Z).fit()
            
            if self.response == 'linear':
                self.y_hat = self.est_model.params[0]*T
            else:
                self.y_hat = self.est_model.params[0]
        elif self.est_id in ['NPDML','LinearDRIV']:
            if len(Z.shape)>1:
                Z = Z[:,-1]
            self.est_model.fit(Y=Y, T=T, Z = Z,W = C)
            self.y_hat = self.est_model.ate()
        elif self.est_id in ['DML','DRIV','Ortho']:
            self.est_model.fit(Y=Y, T=T, Z = Z,W = C)
            self.y_hat = self.est_model.ate()
        elif self.est_id in ['Poly_tsls','KernelIV']:
            self.y_hat= self.est(y=Y,t=T,z=Z,C=C,)

    def effect_estimate(self):
        return self.y_hat

class ce_estimator_report:
    def __init__(self,est_id,treatment_type,response):
        self.est_id = est_id
        self.est = Est[est_id]
        self.treatment_type = treatment_type
        self.response = response
        if est_id == ['tsls', 'IVGMM','Poly_tsls','KernelIV']:
            pass
        elif est_id in ['LinearDRIV']:
            self.est_model = self.est(discrete_treatment=(self.treatment_type=='b'),discrete_instrument=False)
        elif est_id in ['DML','DRIV']:
            self.est_model = self.est(discrete_treatment=(self.treatment_type=='b'),discrete_instrument=False)
        elif est_id == 'Ortho':
            self.est_model = self.est(projection=False, discrete_treatment=(self.treatment_type=='b'), discrete_instrument=False)
        elif est_id == 'NPDML':
            self.est_model = self.est(discrete_treatment=(self.treatment_type=='b'),discrete_instrument=False,model_final=StatsModelsLinearRegression())

    def fit(self,Z,T,Y,C=None):
        
        if self.est_id == 'tsls' or self.est_id == 'IVGMM':
            self.est_model = self.est(dependent = Y, exog = C,endog=T,instruments=Z).fit()
            
            self.y_hat = self.est_model.params[0]
        elif self.est_id in ['NPDML','LinearDRIV']:
            if len(Z.shape)>1:
                Z = Z[:,-1]
            self.est_model.fit(Y=Y, T=T, Z = Z,W = C)
            self.y_hat = self.est_model.ate()
        elif self.est_id in ['DML','DRIV','Ortho']:
            self.est_model.fit(Y=Y, T=T, Z = Z,W = C)
            self.y_hat = self.est_model.ate()
        elif self.est_id in ['Poly_tsls','KernelIV']:
            self.y_hat= np.mean(self.est(y=Y,t=T,z=Z,C=C,))

    def effect_estimate(self):
        return self.y_hat
        
            
