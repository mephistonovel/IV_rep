import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import os
from collections import Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm
from urllib.request import urlretrieve
import gzip
class Generator():
    def __init__(self):
        pass
    
    def generate_synthetic(self, n_samples, dim_D1, dim_D2, dim_Z, dim_C, treatment_type, response_func, seed, within_dependency=False, interaction=False, random=False, true_effect = 3):

        if not random:
            np.random.seed(seed) 
        data = {}        
        if (not within_dependency) and (not interaction):

            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
            function_sampler = GaussianProcessRegressor(kernel=kernel, random_state=0)

            e_D1 = np.random.multivariate_normal([0]*dim_D1, 10*np.eye(dim_D1), n_samples)
            for idx, column in tqdm(enumerate(e_D1.T)):
                data[f'D1_{idx}'] = function_sampler.sample_y(column.reshape(-1,1), random_state=idx).reshape(-1)

            D1 = pd.DataFrame(data)
            e_Z = np.random.multivariate_normal([0]*dim_Z, 0.01*np.eye(dim_Z), n_samples)

            Z = function_sampler.sample_y(D1, dim_Z, random_state=idx+1) + e_Z
            forx = {}
            for i, column in enumerate(Z.T):
                data[f'Z{i}'] = Z[:,i]
       

            C = np.random.multivariate_normal([0]*dim_C, np.eye(dim_C), n_samples)
            fory = {}
            for i, column in enumerate(C.T):
                data[f'C{i}'] = C[:,i]
                forx[f'C{i}'] = C[:,i]
                fory[f'C{i}'] = C[:,i]

            U = np.random.normal(0, 1, n_samples)

            e_D2 = np.random.multivariate_normal([0]*dim_D2, 0.25*np.eye(dim_D2), n_samples)
            pre_D2 = C @ np.random.rand(dim_C, dim_D2)+ e_D2
            D2 = pre_D2[:,:]
            for i, column in enumerate(pre_D2.T):
                flag = np.random.binomial(1,0.5)
                if flag == 1:
                    D2[:,i] = column + U
                data[f'D2_{i}'] = D2[:,i].reshape(-1)
                forx[f'D2_{i}'] = D2[:,i].reshape(-1)
                fory[f'D2_{i}'] = D2[:,i].reshape(-1)
        elif (within_dependency) and (not interaction):
            '''within dep in D1 and D2'''    
            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
            function_sampler = GaussianProcessRegressor(kernel=kernel, random_state=0)

            e_D1 = np.random.multivariate_normal([0]*dim_D1, 10*np.eye(dim_D1)+(np.zeros((dim_D1,dim_D1))+5-5*np.eye(dim_D1)), n_samples)
            for idx, column in tqdm(enumerate(e_D1.T)):
                data[f'D1_{idx}'] = function_sampler.sample_y(column.reshape(-1,1), random_state=idx).reshape(-1)

            D1 = pd.DataFrame(data)
            e_Z = np.random.multivariate_normal([0]*dim_Z, 0.01*np.eye(dim_Z), n_samples)

            Z = function_sampler.sample_y(D1, dim_Z, random_state=idx+1) + e_Z
            forx = {}
            for i, column in enumerate(Z.T):
                data[f'Z{i}'] = Z[:,i]
       

            C = np.random.multivariate_normal([0]*dim_C, np.eye(dim_C), n_samples)
            fory = {}
            for i, column in enumerate(C.T):
                data[f'C{i}'] = C[:,i]
                forx[f'C{i}'] = C[:,i]
                fory[f'C{i}'] = C[:,i]

            U = np.random.normal(0, 1, n_samples)

            e_D2 = np.random.multivariate_normal([0]*dim_D2, 0.25*np.eye(dim_D2)+(np.zeros((dim_D2,dim_D2))+0.125-0.125*np.eye(dim_D2)), n_samples)
            pre_D2 = C @ np.random.rand(dim_C, dim_D2)+ e_D2
            D2 = pre_D2[:,:]
            for i, column in enumerate(pre_D2.T):
                flag = np.random.binomial(1,0.5)
                if flag == 1:
                    D2[:,i] = column + U
                data[f'D2_{i}'] = D2[:,i].reshape(-1)
                forx[f'D2_{i}'] = D2[:,i].reshape(-1)
                fory[f'D2_{i}'] = D2[:,i].reshape(-1)
        elif (within_dependency) and (interaction):
            '''interaction between D1 and D2'''    
            kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
            function_sampler = GaussianProcessRegressor(kernel=kernel, random_state=0)
            UD1D2 = np.random.randn(n_samples)
            
            e_D1 = np.random.multivariate_normal([0]*dim_D1, 10*np.eye(dim_D1), n_samples)
            for idx, column in tqdm(enumerate(e_D1.T)):
                data[f'D1_{idx}'] = function_sampler.sample_y(column.reshape(-1,1), random_state=idx).reshape(-1)+UD1D2

            D1 = pd.DataFrame(data)
            e_Z = np.random.multivariate_normal([0]*dim_Z, 0.01*np.eye(dim_Z), n_samples)

            Z = function_sampler.sample_y(D1, dim_Z, random_state=idx+1) + e_Z
            forx = {}
            for i, column in enumerate(Z.T):
                data[f'Z{i}'] = Z[:,i]
       

            C = np.random.multivariate_normal([0]*dim_C, np.eye(dim_C), n_samples)
            fory = {}
            for i, column in enumerate(C.T):
                data[f'C{i}'] = C[:,i]
                forx[f'C{i}'] = C[:,i]
                fory[f'C{i}'] = C[:,i]

            U = np.random.normal(0, 1, n_samples)

            e_D2 = np.random.multivariate_normal([0]*dim_D2, 0.25*np.eye(dim_D2), n_samples)
            pre_D2 = C @ np.random.rand(dim_C, dim_D2)+ e_D2
            D2 = pre_D2[:,:]
            for i, column in enumerate(pre_D2.T):
                flag = np.random.binomial(1,0.5)
                if flag == 1:
                    D2[:,i] = column + U + UD1D2
                else:
                    D2[:,i] = column + UD1D2
                    
                data[f'D2_{i}'] = D2[:,i].reshape(-1)
                forx[f'D2_{i}'] = D2[:,i].reshape(-1)
                fory[f'D2_{i}'] = D2[:,i].reshape(-1)


        forx = pd.DataFrame(forx)
        dim = len(forx.columns)

        ZtoX = function_sampler.sample_y(Z, 1, random_state=idx+3)
        if treatment_type == 'b':
            p_x = np.exp(ZtoX + (forx.values @ np.concatenate(np.random.randn(dim, 1))).reshape(-1,1) + U.reshape(-1,1))
            p_x = 1 / (1 + p_x)
            X = np.int64(p_x>0.5)
        elif treatment_type == 'con':
            X = ZtoX + (forx.values @ np.concatenate(np.random.randn(dim, 1))).reshape(-1,1)  + U.reshape(-1,1)
        data['X'] = X.reshape(-1)

        fory = pd.DataFrame(fory)
        if response_func == 'linear':
            g = lambda x: true_effect*x
        elif response_func == 'nonlinear':

            g= lambda x: np.exp(0.5*x)

        
        Y = g(X) + function_sampler.sample_y(fory, 1, random_state=idx+2) + U.reshape(-1,1)


        data['Y'] = Y.reshape(-1)

        data = pd.DataFrame(data)

        data = data.sample(frac=1).reset_index(drop=True)

        split_ratio = 0.7
        split_index = int(len(data) * split_ratio)

        data1 = data[:split_index]
        data2 = data[split_index:]


        data_folder = f'./Data/Syn_{n_samples}_{treatment_type}_{response_func}'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        csv_filename1 = f'Syn_train_{"within" if within_dependency else "nodep"}_{"interaction" if interaction else "nointer"}_{true_effect}.csv'
        csv_filename2 = f'Syn_test_{"within" if within_dependency else "nodep"}_{"interaction" if interaction else "nointer"}_{true_effect}.csv'
        data1.to_csv(os.path.join(data_folder, csv_filename1), index=False)
        data2.to_csv(os.path.join(data_folder, csv_filename2), index=False)

        return 0
        
    
    
    def generate_synthetic_highdim(self, n_samples, dim_D2, dim_C, treatment_type, response_func, seed, random=False, true_effect = 3):
        url = 'http://yann.lecun.com/exdb/mnist/'
        files = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']
        data_folder = f'./Data/Syn_MNIST'
        data = {}
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        for file in files:
            if file not in os.listdir(data_folder):
                urlretrieve(url + file, os.path.join(data_folder, file))
                print("Downloaded %s to %s" % (file, data_folder))

        with gzip.open(os.path.join(data_folder, files[0]) )as f:
            tmp1 = np.frombuffer(f.read(), 'B', offset=16).reshape(-1, 784).astype('float32') / 255

        with gzip.open(os.path.join(data_folder, files[2]) )as f:
            tmp2 = np.frombuffer(f.read(), 'B', offset=16).reshape(-1, 784).astype('float32') / 255

        D = np.concatenate((tmp1,tmp2))
        D = D[:n_samples, :]
        kernel = 10 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
        function_sampler = GaussianProcessRegressor(kernel=kernel, random_state=0)
        U = np.random.normal(0, 1, n_samples)
        C = np.random.multivariate_normal([0]*dim_C, np.eye(dim_C), n_samples)
        e_D2 = np.random.multivariate_normal([0]*dim_D2, 0.25*np.eye(dim_D2), n_samples)
        pre_D2 = C @ np.random.normal(0, 0.1, (dim_C, dim_D2))+ e_D2
        D2 = pre_D2[:,:]
        for i, column in enumerate(pre_D2.T):
                flag = np.random.binomial(1,0.5)
                if flag == 1:
                        D2[:,i] = column + U

        D[:, :dim_D2] += D[:, :dim_D2] + D2

        for idx, column in enumerate(D.T):
                data[f'D_{idx}'] = column.reshape(-1)


        with gzip.open(os.path.join(data_folder, files[1])) as f:
            tmpz1 = np.frombuffer(f.read(), 'B', offset=8)

        with gzip.open(os.path.join(data_folder, files[3])) as f:
            tmpz2 = np.frombuffer(f.read(), 'B', offset=8)

        Z = np.concatenate((tmpz1,tmpz2))
        Z = Z[:n_samples]
        data['Z'] = Z
        ZtoX = function_sampler.sample_y(Z.reshape(-1,1), 1, random_state=idx+3)
        forx = np.concatenate((D[:,:dim_D2], C), axis=1)
        dim = dim_D2 + dim_C
        if treatment_type == 'b':
            p_x = np.exp(ZtoX + (forx @ np.random.normal(0, 0.1, (dim, 1))).reshape(-1,1) + U.reshape(-1,1))
            p_x = 1 / (1 + p_x)
            X = np.int64(p_x>0.5)
        elif treatment_type == 'con':
            X = ZtoX + (forx @ np.random.normal(0, 0.1, (dim, 1))).reshape(-1,1)  + U.reshape(-1,1)

        data['X'] = X.reshape(-1)
        if response_func == 'linear':
            g = lambda x: true_effect*x
        elif response_func == 'nonlinear':
            g = lambda x: np.exp(0.5*x)
        fory = np.concatenate((D[:,:dim_D2], C, U.reshape(-1,1)), axis=1)
        Y = g(X) + function_sampler.sample_y(fory, 1, random_state=idx+2) + U.reshape(-1,1)
        data['Y'] = Y.reshape(-1)
        data = pd.DataFrame(data)
        data = data.sample(frac=1).reset_index(drop=True)

        split_ratio = 0.7
        split_index = int(len(data) * split_ratio)

        data1 = data[:split_index]
        data2 = data[split_index:]


        data_folder = f'./Data/Syn_highdim_{n_samples}_{treatment_type}_{response_func}'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        csv_filename1 = f'Syn_train_highdim_d2dim{dim_D2}_dimc{dim_C}_{true_effect}.csv'
        csv_filename2 = f'Syn_test_highdim_d2dim{dim_D2}_dimc{dim_C}_{true_effect}.csv'
        data1.to_csv(os.path.join(data_folder, csv_filename1), index=False)
        data2.to_csv(os.path.join(data_folder, csv_filename2), index=False)





if __name__ == "__main__":
    gen = Generator()
    gen.generate_synthetic_highdim(5000, 100, 50, 'b', 'linear', 0)
  
