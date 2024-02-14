#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error,r2_score, mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

##################

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import colorsys
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
### Dimension reduction and clustering libraries
#import umap.umap_ as umap
from matplotlib.pyplot import cm




############  Create function to read dataframe and do some preprocessing #####################

class create_df:
    '''
        
        - Create a data_df frame
        - get the file path
        - find if the file type is a CSV or excel type
        - return a data frame
        
    '''
    def __init__(self, pathfile):
        self.pathfile = pathfile
        self.filetype = pathfile.split('.')[-1]
        
    
    
    def read_data (self, delimiter_sep=',', **kwargs):
        # read data and create a data_df frame 
        print('read file:', self.pathfile)
        # if data_df is a csv file
        if (self.filetype == 'csv'):
            data_df = pd.read_csv(self.pathfile, delimiter=delimiter_sep, **kwargs)
        
        # if data_df is an excel file 
        if (self.filetype == 'xlsx'):
            data_df = pd.read_excel(self.pathfile, **kwargs )
        
        print('The shape of original data_df:', data_df.shape)
        print()
        return data_df

    
    
    
class preprocessing:
    '''
        - doing the first step in data_df preprocessing 
            - remove duplicate rows
            - sort the rows
            - remove Nan columns
            - remove columns with constant value
            - remove Nan rows
            - remove rows with constant value
            - add some new features
    '''       
    def __init__(self, data_df):
        
        self.data_df = data_df
    
    def drop_duplicate_rows(self):
        
        print('Number of duplicate rows',self.data_df.duplicated().sum() )
        self.data_df = self.data_df.drop_duplicates()
        print('the shape of data_df after remove duplicates:', self.data_df.shape ,'\n')
        
        return self.data_df
        
    def sort_data_df (self, sort_col_list):
        print('sort values w.r.t', sort_col_list,'\n' )
        self.data_df.sort_values(sort_col_list,inplace=True)
        return self.data_df
    
    def remove_columns_with_all_Nan(self):
        # a loop over all columns
        for i in self.data_df.columns:
            # find the columns with all nan
            allnull= self.data_df[i].isnull().all()
            if allnull:
                print( f'drop the columns -- {i} -- with nan value ')
                # drop column with all nan
                self.data_df=self.data_df.drop([i], axis=1)
    
        return self.data_df
            
        
    def remove_columns_with_constant_value(self, exceptation_cols=[]):
    ## it does not consider nan for example if the unique values are [nan, 0] it is considered as constant value
    
        for i in self.data_df.columns:
            # loop over all columns
            # find unique values for each columns
            len_unique = len(self.data_df[i].dropna().unique())
            if len_unique == 1:
                #if the length of the unique is 1 
                # check if is not in the list of exception 
                # drop the column with only one unique value
                if i not in (exceptation_cols):
                    print( f'drop the columns -- {i} -- with constant value of {self.data_df[i].unique()}')
                    self.data_df=self.data_df.drop([i], axis=1)
    
        return self.data_df
    
    def remove_rows_with_all_Nan (self):
        self.data_df = self.data_df.dropna(axis = 0, how = 'all')
        print('the shape of data after removing the rows with all Nan:', self.data_df.shape , '\n')
        
        return self.data_df
    
    def remove_rows_with_all_zeros (self):
        self.data_df.loc[(self.data_df!=0).any(axis=1)]
        print('the shape of data after removing the rows with all Zero:', self.data_df.shape, '\n')
     
        return self.data_df
    
        
    def create_new_colums (self, cols_list_to_sum: list[str], new_col_name: str ):
        # add new feature = sum of the all types of elements

        self.data_df[new_col_name]= self.data_df[cols_list_to_sum].sum(axis=1)
        return self.data_df
    
    
    
def preprocessing_func (data_df, sort_col_list):
    # creat a new class for preprocessing 
    preprocess_class = preprocessing(data_df)
    # sort dataset
    data_df = preprocess_class.sort_data_df(sort_col_list)
    # Remove rows and columns with zero and nan entries and equal values

    data_df = preprocess_class.remove_columns_with_all_Nan()

    data_df = preprocess_class.remove_columns_with_constant_value()

    data_df = preprocess_class.remove_rows_with_all_Nan()

    data_df = preprocess_class.remove_rows_with_all_zeros()

    data_df = preprocess_class.drop_duplicate_rows()
    
    return data_df


################### Get the outliers and drop the samples ########################
class outliers():
    '''
        define the outliers using std, iqr
        - create a dict for each columns with upper and lower limits
    '''
    def __init__(self, data_df):
        self.data_df = data_df
        
        
    def outliers_std(self,col, nstd= 3):
       
        # calculate summary statistics
        data_mean = self.data_df[col].mean()
        data_std = self.data_df[col].std()

        # identify lower and upper limits to detect outliers
        cut_off = data_std * nstd
        lower_std, upper_std = data_mean - cut_off, data_mean + cut_off

    
        return lower_std, upper_std


    def outliers_iqr(self,col, k=1.5):
 
        # calculate interquartile range
        q25 = self.data_df[col].quantile(0.25)
        q75 = self.data_df[col].quantile(0.75)
        iqr = q75 - q25

        # calculate the outlier cutoff
        cut_off = iqr * k
        lower_iqr, upper_iqr = q25 - cut_off, q75 + cut_off

    
        return lower_iqr, upper_iqr

    def find_limits_outlier (self, col_list, nstd=3, k=1.5):
        # the limits of a list of columns for a dataset
       
        dics_limits_std = {}
        dics_limits_iqr = {}
        for col_name in col_list:


           
            lower_std, upper_std = self.outliers_std(col_name, nstd)
            dics_limits_std[col_name]=[lower_std, upper_std]

            lower_iqr, upper_iqr = self.outliers_iqr(col_name,k)
            dics_limits_iqr[col_name]=[lower_iqr, upper_iqr]

     
        return dics_limits_std, dics_limits_iqr
    
    
def drop_outliers_rows_col_seperate (data, dict_list):
    # omit the rows where the boundaries of the column are outside the defined range
    for key in dict_list:
        upper = dict_list[key][1]
        lower = dict_list[key][0]
        if not np.isnan(upper):
            data = data[data[key] <= upper]
        if not np.isnan(lower):
            data = data[data[key] >= lower]
            
 
    return data

def plot_limits_std_iqr (data, dict_limits_std, dict_limits_iqr, figsize=(15,2), plot_std=True, plot_iqr=True):
    # plot the upper and lower limit using std, iqr
    # plot the min value and max value of the parameters
    
    for col in data.columns:
        bins=50
        y = data[col]
        
        
        y_min  = y.min()
        y_max  = y.max()
        
        
        (lower_std, upper_std) = dict_limits_std[col]
        (lower_iqr, upper_iqr) = dict_limits_iqr[col]


        plt.figure(figsize=figsize)
        plt.hist(y, bins=bins,density=False, alpha=0.5, color='b')
        if plot_std:
            plt.axvline(x = lower_std, color = 'r',linestyle='-',alpha=0.5, label = 'std')
            plt.axvline(x = upper_std, color = 'r',linestyle='-',alpha=0.5, label = '')
        if plot_iqr:
            plt.axvline(x = lower_iqr, color = 'g',linestyle='-',alpha=0.5, label = 'iqr')
            plt.axvline(x = upper_iqr, color = 'g',linestyle='-',alpha=0.5, label = '')

        
        plt.axvline(x = y_min, color = 'b',linestyle='--',alpha=0.5, label = 'min value')
        plt.axvline(x = y_max, color = 'b',linestyle='--',alpha=0.5, label = 'max value')

        

        plt.legend()

        plt.xlabel(col, fontsize=12)
        plt.show()           
   




########### some plot functions ######################
def corrmatrix_heatmap (data, method= 'pearson'):
    #Pearson’s correlation coefficient (linear).
    #Spearman’s rank coefficient (nonlinear)
    
    corrMatrix = data.corr(method)
    
    # plot only the half of the matrix
    plt.figure(figsize=(20, 10), dpi=80)

    mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
    sns.heatmap(corrMatrix, annot=True, vmax=1, vmin=-1,cmap='Blues', center=0, mask=mask)
    plt.show()
    
def scatter_plot (data, xcol, ycol, xlabel,ylabel, title, **kwargs):
    x = data[xcol]
    y = data[ycol]
    
    plt.scatter(x,y,**kwargs)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

def plot_loghist(x, bins, **kwargs):
    hist, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    plt.hist(x, bins=logbins, **kwargs)
    plt.xscale('log')




##################  Machine learning #######################

def evaluate_regression (y, y_predict, num_features):
    score=r2_score(y,y_predict)
    ### r2= 1-(SSres / SSmean)
    #### SSres = 1/n(sum (y-y_predict)^2)
    #### SSmean = 1/n(sum (y-y_mean)^2)
    ### The best possible score is 1.0  for r2_score
    ### R2_adjust = 1−[(1−R2)*(n−1)/(n−p−1)], where p is the number of features and n the number of instances.
    
    '''
    R Squared: R Square is the coefficient of determination.
    It tells us how many points fall on the regression line. 
    R-squared usually ranges between 0 for models where the model does not explain the data at all 
    and 1 for models that explain all of the variance in your data.
    R2<0 happens when SSres is greater than SSmean which means that a model does not capture the
    trend of the data and fits to the data worse than using the mean of the target as the prediction.
    
    R-squared increases with the number of features in the model, even if they do not contain any information about the target value at all. Therefore, 
    it is better to use the adjusted R-squared, which accounts for the number of features used in the model. 
    R2_adjust = 1−[(1−R2)*(n−1)/(n−p−1)], where p is the number of features and n the number of instances.
    It is not meaningful to interpret a model with very low (adjusted) R-squared, because such a model basically does not explain much of the variance.
    Any interpretation of the weights would not be meaningful.
    
    Mean Absolute Error: Mean Absolute Error is the absolute difference between
    the actual or true values and the predicted values. The lower the value, 
    the better is the model’s performance. A mean absolute error of 0 means
    that your model is a perfect predictor of the outputs. ( it is a mean )
    
    
    Mean Square Error: Mean Square Error is calculated by taking the average 
    of the square of the difference between the original and predicted values of the data.
    The lower the value, the better is the model’s performance. (It is a variance)
    
    Root Mean Square Error: Root Mean Square Error is the standard deviation of the errors
    which occur when a prediction is made on a dataset. This is the same as Mean Squared Error, 
    but the root of the value is considered while determining the accuracy of the model. 
    The lower the value, the better is the model’s performance. 
    RMSE gives relatively higher weight to large errors. (It is a standrad deviation)
    
    point:
    - RMSE can be troublesome when calculating on different sized test samples
        (As is the case when training a model). RMSE has a tendency to be increasingly 
        larger than MAE as the test sample size increases ([RMSE] ≤ [MAE * sqrt(n)].
        
    - R2 is not good choice to evalute the model. 
    
    '''
    MAE = mean_absolute_error(y,y_predict)
    MSE = mean_squared_error(y,y_predict)
    RMSE = np.sqrt(mean_squared_error(y,y_predict))
    r2_adjusted = 1- ((1-score)*(len(y)-1))/(len(y)-num_features-1)
    
    print('r2 socre is ',score)
    print('r2 adjusted ', r2_adjusted )
    print('MAE ==',MAE)
    print('MSE ==', MSE)
    print('RMSE ==',RMSE)
    #print('Mean Error == ', np.mean(y-y_predict) )

    # put all scores in a dictionary
    scors_dic = {'r2': score,
                 'r2_adjusted': r2_adjusted,
             'MAE': MAE,
             'MSE': MSE, 
             'RMSE': RMSE}
    
    return scors_dic

class modeling:
    
    '''
    - input is the model name
    '''
    def __init__(self, ML_model):
        self.model = ML_model
        
    def model_fit (self, X_train, y_train):
        # fit the model on train data set
        result = self.model.fit(X_train, y_train)
        
        return result
    
    def predict_y (self, X, y):
        ## evalute the model
        y_predict = self.model.predict(X)
        score_dics = evaluate_regression(y, y_predict,X.shape[1] )

        return y_predict, score_dics
    
   
    
    
    def get_coefficient_df (self, X_train, name_col ):
        ## if the model is linear regression we get the coef
       
        try:
            coef_df = pd.DataFrame(self.model.coef_, columns=name_col, index=X_train.columns)
            coef_df.sort_values(name_col,inplace=True)

        except:
            print('the coef is not defined for',type(self.model).__name__)
            coef_df = None
            
        return coef_df
    
    def get_feature_importance (self,X_train):
        # feature importance
        try:
            importances = self.model.feature_importances_
            # sort them
            indices = np.argsort(importances)
            # get the feature name
            features = X_train.columns

            # barchar
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], color='g', align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel('Relative Importance')
            plt.show()
        except:
            print('there is no feature imprtances for', type(self.model).__name__)
         
        

    def plot_predict_actual (self,y, y_predict, title, x_label,y_label):
        ## plot the actual and predict values and linear line
        plt.figure(figsize=(5,5))
        plt.scatter(y, y_predict)

        p1 = max(max(y_predict), max(y))
        p2 = min(min(y_predict), min(y))
        plt.plot([p1, p2], [p1, p2], 'r--')
        plt.title(title,fontsize=15 )
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=15)
        #plt.axis('equal')
        plt.show()

    

def apply_model (ML_model, X, y):
    y_predict = ML_model.predict(X)
    score_dics = evaluate_regression(y, y_predict,X.shape[1] )
    
    return score_dics





def normalization (data, drop_col_list = [], method='MinMax'):
    # fit scaler on training data
    ## X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    ###X_scaled = X_std * (max - min) + min
   
    if method == 'MinMax':
        norm = MinMaxScaler()
    elif method == 'Standard':
        norm = StandardScaler()

    # not all columns need to be normalized
    # drop the columns which should be excluded from normalization
    data_ = data.drop(drop_col_list, axis=1)
    data_drop = data[drop_col_list]

    # fit the rest of data
    norm.fit(data_)
    # transform training data
    data_norm = norm.transform(data_)
    data_norm_df = pd.DataFrame(data_norm, columns = data_.columns)
    # The columns that were excluded from normalization will be added again.
    data_norm_df  = data_norm_df.merge(data_drop, left_index=True, right_index=True)       
    return norm, data_norm_df







