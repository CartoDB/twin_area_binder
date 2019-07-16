from sklearn import preprocessing
from ppca import PPCA
import pandas as pd

# def distance_between(data, origin_cell_id):
#     u = data.as_matrix()
#     v = np.array([data.loc[origin_cell_id].as_matrix()])
#     data['distance'] = pairwise_distances(u, v, metric='euclidean')
#     return data

def compute_pca(data, predictors, how = 'pca', what = 'factors', n_components = 1, use_corr = True):
    
    data[predictors] = data[predictors].astype('float64')
    X = data[predictors].values
    if(use_corr == True):
        ## If PCA are computed using the correlation matrix --> standardize the data (zero mean, unit std.)
        scaler = preprocessing.StandardScaler()
        X_std = scaler.fit_transform(X)
    else:
        ## If PCA are computed using the covariance matrix --> center the data only (zero mean)
        X_mean = np.mean(X, axis = 0)
        X_std = X - X_mean

    if(how == 'pca'):
        pca = PCA(n_components)
        pca.fit(X_std)
        factors = pca.transform(X_std)
        explained_variance = pca.explained_variance_ratio_
        Xhat_std = pca.inverse_transform(factors)
        if(use_corr == True):
            Xhat = scaler.inverse_transform(Xhat_std)
        else:
            Xhat = Xhat_std + X_mean

    if(how == 'ppca'):
        ppca = PPCA()
        ppca.fit(X_std, n_components)
        factors = ppca.transform()
        explained_variance = ppca.var_exp
        Xhat_std = ppca.reconstruct()
        if(use_corr == True):
            Xhat = scaler.inverse_transform(Xhat_std)
        else:
            Xhat = Xhat_std + X_mean

    if(what != 'recon'):
        pca_columns = []
        for i in range(factors.shape[1]):
            pca_columns.append('pca_{}'.format(i))
            data['pca_{}'.format(i)] = factors[:,i]
        return list([data, explained_variance])
    else:
        rec_data = pd.DataFrame(Xhat)
        return rec_data
