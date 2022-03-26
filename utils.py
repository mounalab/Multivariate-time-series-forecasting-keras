import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import keras.backend as K

def mape(y_true, y_pred):
    import keras.backend as K
    """
    Returns the mean absolute percentage error.
    For examples on losses see:
    https://github.com/keras-team/keras/blob/master/keras/losses.py
    """
    return (K.abs(y_true - y_pred) / K.abs(y_pred)) * 100
    #diff = K.abs(y_true - y_pred) / K.abs(y_true)
    #return 100. * K.mean(diff)#, axis=-1)

def smape(y_true, y_pred):
    import keras.backend as K
    """
    Returns the Symmetric mean absolute percentage error.
    For examples on losses see:
    https://github.com/keras-team/keras/blob/master/keras/losses.py
    """
    return 100*K.mean(K.abs(y_pred - y_true) / ((K.abs(y_true) + K.abs(y_pred))), axis=-1)
    #Symmetric mean absolute percentage error
    #return 100 * K.mean(K.abs(y_pred - y_true) / (K.abs(y_pred) + K.abs(y_true)))#, axis=-1)

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mae(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def mase(y_true, y_pred):

    sust = K.mean(K.abs(y_true[:,1:] - y_true[:,:-1]))
    diff = K.mean(K.abs(y_pred - y_true))

    return diff/sust

def coeff_determination(y_true, y_pred):

    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# convert time series to 2D data for supervised learning
def series_to_supervised(data, train_size=0.5, n_in=1, n_out=1, target_column='target', dropnan=True, scale_X=True):

    df = data.copy()

    # Make sure the target column is the last column in the dataframe
    df['target'] = df[target_column] # Make a copy of the target column
    df = df.drop(columns=[target_column]) # Drop the original target column

    target_location = df.shape[1] - 1 # column index number of target

    # ...X
    #X = df.iloc[:, :target_location]
    X = df.iloc[:,:]

    # ...y
    y = df.iloc[:, [target_location]]

    # Scale the features
    if scale_X:
        #col_names=['target']
        #features = X[col_names]
        features = X[X.columns]
        scalerX = MinMaxScaler().fit(features.values)
        features = scalerX.transform(features.values)

        #X['target'] = features
        X[X.columns] = features

    #n_vars_x = X.shape[1]
    x_vars_labels = X.columns
    y_vars_labels = y.columns

    x_cols, x_names = list(), list()
    y_cols, y_names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        x_cols.append(X.shift(i))
        x_names += [('%s(t-%d)' % (j, i)) for j in x_vars_labels]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        y_cols.append(y.shift(-i))
        if i == 0:
            y_names += [('%s(t)' % (j)) for j in y_vars_labels]
        else:
            y_names += [('%s(t-%d)' % (j, i)) for j in y_vars_labels]

    # put it all together
    x_agg = pd.concat(x_cols, axis=1)
    x_agg.columns = x_names

    y_agg = pd.concat(y_cols, axis=1)
    y_agg.columns = y_names

    agg=pd.concat([x_agg,y_agg], axis=1)
    agg.columns = x_names + y_names
    #print(agg)


    # drop rows with NaN values
    if dropnan:
        x_agg.dropna(inplace=True)
        y_agg.dropna(inplace=True)

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    """
    diff = y_agg.shape[0] - x_agg.shape[0]
    idx = [i for i in range(0, diff)]
    y_agg = y_agg.drop(df.index[idx])"""

    nf = X.shape[1]
    xx = agg.iloc[:,:n_in*nf]
    yy = agg.iloc[:,-n_out:]

    split_index = int(xx.shape[0]*train_size) # the index at which to split df into train and test

    # ...train
    X_train = xx.iloc[:split_index, :]
    y_train = yy.iloc[:split_index, ]

    # ...test
    X_test = xx.iloc[split_index:, :] # original is split_index:-1
    y_test = yy.iloc[split_index:, ] # original is split_index:-1

    return X_train, y_train, X_test, y_test, scale_X
