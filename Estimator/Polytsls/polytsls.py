from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

def poly_tsls(y,t,z,C=None):
   
    params = dict(poly__degree=range(1, 3), ridge__alpha=np.logspace(-5, 5, 11))
    pipe = Pipeline([('poly', PolynomialFeatures()), ('ridge', Ridge())])
    stage_1 = GridSearchCV(pipe, param_grid=params, cv=5)
    # stage_1.fit(np.concatenate([data['Z1'], data['Z2']], axis=1), data['T'])
    if C is not None:
        reg = np.concatenate([z,C],axis=1)
        stage_1.fit(np.concatenate([z,C], axis=1), t)
        t_hat = stage_1.predict(np.concatenate([z,C], axis=1))
    else:
        if len(z.shape)<2:
            z = z.reshape(-1,1) 
        reg = z

        
        stage_1.fit(reg, t)
        t_hat = stage_1.predict(z)

    pipe2 = Pipeline([('poly', PolynomialFeatures()), ('ridge', Ridge())])
    stage_2 = GridSearchCV(pipe2, param_grid=params, cv=5)
    
    if C is not None:
        regx = np.concatenate([t_hat.reshape(-1,1),C],axis=1)
        stage_2.fit(regx, y)
        y_hat = stage_2.predict(np.concatenate([t.reshape(-1,1),C],axis=1))
        # y_hat0 = stage_2.predict(np.concatenate([np.zeros((y.shape[0],1)),C],axis=1))
        
    else:
        regx = t_hat.reshape(-1,1)
        stage_2.fit(regx, y)
        y_hat = stage_2.predict(t.reshape(-1,1))
        # y_hat0 = stage_2.predict(np.zeros((y.shape[0],1)))

    return y_hat