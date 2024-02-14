#!/usr/bin/env python
# coding: utf-8



## in this module some functions are defiend which are used in the AIPI project
## this function will go to the utlis_aipi modul



import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split



# the metadata is not calculated for these columns
COLUMNS_DROP= []


# create a dic to show which scrap group togather
# key is the col name for grouped scrap and value is a list which shows which scraps are put together.
dict_group_scrap = {'scrap1':['scrap1','scrap3']}

def scrap_typ1_group_func (scrap_df,dict_group_scrap):
    for key, value in dict_group_scrap.items():
        # all of the type scraps which sum up may not exist always. 
        # get only the scraps in this dataset
        scrap_cols= list(set(value).intersection(scrap_df.columns))
       
        # sum up the cols
        sum_scraps = scrap_df[scrap_cols].sum(axis=1)
        # remove the cols list 
        scrap_df.drop(scrap_cols,axis=1, inplace=True)
        # add the sum cols
        scrap_df.insert(loc=len(scrap_df.columns)-1, column=key, value=sum_scraps )
        
    return scrap_df


def normalized_scrap_func(scrap_df, extend_list_to_drop, tot_weight_col):
    # 

    # divided scrap to scrap_tot per heat (some columns need to exclude)
    # exclude columns list:
    cols_drop_list = list(COLUMNS_DROP)
    cols_drop_list.extend(extend_list_to_drop)
    exclude_columns = list(set(cols_drop_list).intersection(scrap_df.columns))
    scrap_fraction = scrap_df.loc[:, ~scrap_df.columns.isin(exclude_columns)].div(scrap_df[tot_weight_col], axis=0)
    print('The columns of the fraction dataset')
    display(scrap_fraction.columns)
    # create a new copy of fraction 

    scrap_norm_df= scrap_fraction.copy()


    # merge the excluded first columns
    scrap_norm_df = scrap_df[exclude_columns].merge(scrap_norm_df, left_index=True, right_index=True)
    return scrap_norm_df

def get_features_list_dataset (data_df, extend_list):
    # get the list of scrap
    # list(aipi_utlis.COLUMNS_DROP_METADATA) the reason for list in this line remove the effect to the original list
    cols_drop_list = list(COLUMNS_DROP)
    cols_drop_list.extend(extend_list)
    cols_to_drop = list(set(cols_drop_list).intersection(data_df.columns))
    feature_list = data_df.columns.drop(cols_to_drop).to_list()
    print('the list of features for the dataset\n',feature_list)
    
    return feature_list



############ create the dataset  #####################
class dataset:
    '''
     find the data set for each element to Machine learning
     
    '''
    def __init__(self,y_element):
        # for what kind of element we need to get the input data
        self.y_element = y_element
     
    
    def get_input (self,list_of_data):
        input_df = pd.DataFrame()
        for data in list_of_data:
            if input_df.empty:
                input_df = data.copy()
            else:
                input_df = input_df.merge(data, on='heat',how='inner')
            
        print('the shape of data set with outliers:', input_df.shape)
        return input_df
    
    def get_X_Y (self, input_df,list_of_features, list_of_y=[]):
        ## if we need to add another element for y vector
        X_cols = list_of_features
        X = input_df[X_cols]
        Y = input_df[['heat',self.y_element]+list_of_y]
        return X, Y
    
    def test_train (self,X,Y, test_size, **kwargs):
        # if we do not split the dataset to train and test
        if test_size == 0.0 :
            X_train = X.copy()
            Y_train = Y.copy()
            print('the shape train dataset', X_train.shape)

            ### X_* _heat includes heat column 
            X_train_heat = X_train.copy()
            Y_train_heat = Y_train.copy()

          
            ## drop heat column in X_
            X_train = X_train.drop(['heat'],axis=1)
            Y_train = Y_train.drop(['heat'],axis=1)
           
            return X_train_heat, Y_train_heat, X_train, Y_train
            
        else:
            # split data set to train and test
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, **kwargs)
            print('the shape train dataset', X_train.shape)
            print('the shape test dataset', X_test.shape)

            # reset index 
            X_train.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)
            Y_train.reset_index(drop=True, inplace=True)
            Y_test.reset_index(drop=True, inplace=True)

            ### X_* _heat includes heat column 
            X_train_heat = X_train.copy()
            Y_train_heat = Y_train.copy()

            X_test_heat = X_test.copy()
            Y_test_heat = Y_test.copy()

            ## drop heat column in X_
            X_train = X_train.drop(['heat'],axis=1)
            Y_train = Y_train.drop(['heat'],axis=1)
            X_test = X_test.drop(['heat'],axis=1)
            Y_test = Y_test.drop(['heat'],axis=1)

            return X_train_heat, Y_train_heat, X_test_heat, Y_test_heat, X_train, Y_train, X_test, Y_test
    
    

    
            
def create_X_Y_func (y_element, list_features, extend_list, list_dataset_model,test_size,list_of_y=[], **kwargs):
    # create a class for the dataset
    dataset_y_element = dataset(y_element=y_element)
    input_y_element_df=dataset_y_element.get_input(list_dataset_model)
    # list of features to include in the ML model
    list_of_features_y_element = list(list_features)
    # add some features like heat
    list_of_features_y_element.extend(extend_list)
    print(list_of_features_y_element)
    # get the X and Y
    X_y_element,Y_y_element = dataset_y_element.get_X_Y (input_y_element_df, list_of_features_y_element, list_of_y)
    # divided to train and test
    # the dataset with suffix _heat means there is a column name heat which reresents the heat number
    
    return dataset_y_element.test_train (X_y_element,Y_y_element, test_size, **kwargs)
    
    
        



def add_new_feature(data_df_name, data_df):
    if 'data_input2'.lower() in data_df_name.lower():
        # Total oxygen, coke, and gas
        data_df['new_feature_data_input2']= data_df[[col for col in data_df.columns if col.lower().startswith('fea1')]].sum(axis=1)

        
    
    
    return data_df


def get_anomaly_conditions (data_df_name):
    column_dir_anomaly = {}
    
    if 'data_input2'.lower() in data_df_name.lower():        
        column_dir_anomaly ['fea1'] = {'upper':True, 'lower':False} 
    
    return column_dir_anomaly
        
        
def get_outlier_conditions (data_df_name):
    column_dir_outlier = {}
    
    if 'data_input2'.lower() in data_df_name.lower():       
        column_dir_outlier ['fea1'] = {'upper':False, 'lower':True}  
  
    return column_dir_outlier
    




def find_typ2_heat (scrap_df):
    
    
    # create new dataframe for typ2 steel
    # and drop unnecessary features
    scrap_typ2_df = scrap_data[scrap_data['typ']=='typ1']
    try:
        typ2_heats = scrap_typ2_df['heat'].unique().tolist()
    except:
        typ2_heats = []

    print('Nr of typ2 steel heats:', len(typ2_heats), '\n')

    # create new dataframe for typ1
    
    scrap_typ1_df = scrap_data[scrap_data['typ']=='typ2']

    typ1_heats = scrap_typ1_df['heat'].unique().tolist()
    print('Nr of typ1 heats:', len(typ1_heats))
    
    return heat_list_inconsistency, typ2_heats, typ1_heats

    

    

