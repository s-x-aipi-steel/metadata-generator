#!/usr/bin/env python
# coding: utf-8


# # Import modules



import pandas as pd
import numpy as np

import os
from datetime import datetime

import pathlib

# load ML models
import pickle

# Self-made modules
import code.utlis as utlis # general functions
import code.myconfig as myconfig# Config file
import code.constants as constants


import code.metadata_utlis as metadata_utlis # some function for creating met'adata

import code.aipi_utlis as aipi_utlis # some functions which are used in AIPI project




## for API
import requests
import json
import code.myconfig as myconfig # Config file

SERVICE_ROOT_URL = myconfig.service_url
HEADERS = {'access-token': myconfig.auth_token}



# the data needs to be saved in short and long format
raw_data_period = input('Enter shorter or longer time period:')

__file__ = os.path.join(os.getcwd())
# where the raw data were saved
data_root_dir_rawdata = os.path.join(pathlib.Path(__file__).parent.resolve(), "Data_API", "Raw_data_API",raw_data_period )
# where the metadata will be saved
data_root_dir_metadata = os.path.join(pathlib.Path(__file__).parent.resolve(), "Data_API", "Meta_data_API", raw_data_period)

# where the ML models were saved
data_root_dir_ML = os.path.join(pathlib.Path(__file__).parent.resolve(),'modeling', 'ML_models')

data_root_dir_train_model = os.path.join(pathlib.Path(__file__).parent.resolve(), "Data_API", "Meta_data_API")



print(data_root_dir_rawdata)
print(data_root_dir_metadata)
print(data_root_dir_ML)



def metadata_extract_transformation (process_data_name, data_root_dir_rawdata, dir_name, typ1_heats,typ2_heats, scrap_raw_df):
    '''
    input:
        process data 
        typ1_heats : the typ1 heat list
        typ2_heats : the list of typ2 heats
        scrap_raw_df : some of input files do not have time feature. This missing parameter can be 
                        defiend with the help of scrap input data features    
        
        
    '''

        
    pathfile = data_root_dir_rawdata+'\\'+dir_name+'\\'+process_data_name
    # read process data using class create_df
    process_class = utlis.create_df(pathfile)
    process_raw_df = process_class.read_data()
    #Add some additional functions if required for certain data entries  
    process_raw_df = aipi_utlis.add_new_feature(process_data_name, process_raw_df)
    
    # time doesn't exists in all raw data. It will be added using scrap_input data 
    start_date = scrap_raw_df['time'].min(skipna=True)
    end_date = scrap_raw_df['time'].max(skipna=True)
    if 'time' not in process_raw_df.columns:
        process_raw_df = pd.merge(process_raw_df, scrap_raw_df[['heat','time']], on='heat', how='left')
        
   
        
    ######################## Transformation metadata ###################################################
    # create a class for transformation metadata
    process_transformation = metadata_utlis.metadata_transformation_class(process_raw_df,start_date, end_date, typ1_heats, typ2_heats)
    # the metadata now create only for typ1 steel type
    process_typ1 = process_transformation.return_typ1_df()
    process_total = process_transformation.return_total_df()

    # the first metadata is information about heat
    process_heat_info=process_transformation.Metadata_heats_info()
    
    # the next metadata is statistical analysis of the process data. but not for all of the columns
    # list of columns which do not need to get the statistical and zeros 
    cols_drop_list = aipi_utlis.COLUMNS_DROP
    cols_to_drop = list(set(cols_drop_list).intersection(process_total.columns))
   

    # statistical analysis for process data
    process_statistical = process_transformation.Metadata_statistical(columns=process_total.columns.drop(cols_to_drop))

        
    # Number of zeros for each period
    if 'data_input2'.lower() in process_data_name.lower(): 
       
        process_zeros = process_transformation.Metadata_number_Nans(columns=process_total.columns.drop(cols_to_drop))
    else: 
        # in the other dataset number of zeros are really zero means the matrial with zero value is not used in the process. 
        process_zeros = process_transformation.Metadata_number_zeros(columns=process_total.columns.drop(cols_to_drop))
        # sometimes there is also Nan
        process_nans = process_transformation.Metadata_number_Nans(columns=process_total.columns.drop(cols_to_drop))
        process_zeros = pd.merge(process_zeros,process_nans, on=['Start_date','End_date'])

    # merge the transformation metadata together and create a new dataset
    process_metadata_transformation = [process_heat_info, process_statistical, process_zeros]
    # set the start and End date as index to use concat function
    dfs = [df.set_index(['Start_date','End_date']) for df in process_metadata_transformation]
    #back index to column
    process_metadata_transformation_df = pd.concat(dfs, axis=1).reset_index()
    
    # return metadata for transformation and process_typ1, and start and end date for this period
    return process_metadata_transformation_df, process_typ1, start_date, end_date



def metadata_extract_exploration (process_data_name, process_typ1, start_date, end_date):
    # exploration data is only for typ1 data
    ################################### Exploration metadata ##################
    # create a class for exploration  metadata
    process_exploration = metadata_utlis.metadata_exploration_class(process_typ1,start_date, end_date)

    # the metadata now create only for typ1 steel type
    # the first metadata is information about heat
    process_heat_info=process_exploration.Metadata_heats_info()
    
    # the next metadta is defined the outliers and anomalies for some process data. but not for all of the columns
    columns_dir_outliers = aipi_utlis.get_outlier_conditions (process_data_name)
    columns_dir_anomalies = aipi_utlis.get_anomaly_conditions (process_data_name)
 

    # get outliers for process data
    process_outliers = process_exploration.Metadata_outliers(columns_dir=columns_dir_outliers,method='std', nstd=4)
    
    
    # get anomalies for process data
    process_anomalies = process_exploration.Metadata_anomalies(columns_dir=columns_dir_anomalies,method='std', nstd=4)
    

    # merge the transformation metadata together and create a new dataset
    process_metadata_exploration = [process_heat_info, process_outliers, process_anomalies]
    # set the start and End date as index to use concat function
    dfs = [df.set_index(['Start_date','End_date']) for df in process_metadata_exploration]
    process_metadata_exploration_df = pd.concat(dfs, axis=1).reset_index()
    
    return process_metadata_exploration_df


def find_ML_model_period (start_date_data, end_date_data, model_name, data_root_dir_ML):
    '''
    start_date_data is the start date for this data periode
    end_date_data is the end date for this data periode
    model name the type of model there are different model
    '''
    
    # get the list of all saved model
    list_of_models = os.listdir(data_root_dir_ML) # this line gives all files in directory
    model_list = [model for model in list_of_models if model.endswith('.sav')] # give only saved model with pickle
    # create a list of all saved model
    list_models_dict = []
    for model in model_list:
        # get the model path
        path_save_model_priod = data_root_dir_ML + '\\'+ model
        # load the model
        loaded_data = pickle.load(open(path_save_model_priod, 'rb'))
        # put models as a dict to the list
        model_info = {'model':model, # the name of the file
                    'model_name':loaded_data['model_metadata']['model_name'],
                    'model_start_date':loaded_data['model_metadata']['time_start'],
                    'model_end_date':loaded_data['model_metadata']['time_end']}
        
      
        
        
        
        
        list_models_dict.append(model_info)
    
    
    # Convert strat and end periodes dates to datetime objects
    date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
    start_date_data = datetime.strptime(start_date_data,date_format)
    end_date_data = datetime.strptime(end_date_data,date_format)

    
    best_match = None
    min_difference = float('inf')
    
    for model in list_models_dict: 
        if model['model_name'] == model_name:
            # Convert model dates to datetime objects
            date_format = '%Y-%m-%d'

            model_start_date = datetime.strptime(model['model_start_date'], date_format)
            model_end_date = datetime.strptime(model['model_end_date'], date_format)
            # Check if the entry meets the desired conditions
            if (start_date_data  >= model_end_date):
                # Calculate the difference between desired_end_date and model_end_date
                difference = abs((start_date_data - model_end_date).days)

                # Update the best match if the current entry has a closer end date
                if difference < min_difference:
                    min_difference = difference
                    best_match = model['model']
        
    return best_match
    



def metadata_extract_modeling(all_process_typ1_data_dict,start_date, end_date, y_element, path_save_model ):
    
    scrap_typ1_df = all_process_typ1_data_dict['scrap_input']
    # get the start and end of data set:
    # create a dataframe to put all infos about heats
    metadata_modeling_df = pd.DataFrame() 
        
    ## start and end of the heat time
     
    metadata_modeling_df.at[0,'Start_date'] = start_date
    metadata_modeling_df.at[0,'End_date'] = end_date
        
    # get the number of heats
    metadata_modeling_df.at[0,'Nr_heats'] = int(len(scrap_typ1_df['heat']))
    
    #####################################
    input3_typ1_df = all_process_typ1_data_dict['data_input3']
        
    #### scrap data
    # preprocess step
    scrap_typ1_df = utlis.preprocessing_func (scrap_typ1_df, sort_col_list=['time','heat'])
    # group scraps 
    scrap_typ1_df = aipi_utlis.scrap_typ1_group_func (scrap_typ1_df,aipi_utlis.dict_group_scrap)
    # normalized step
    scrap_typ1_norm_df = aipi_utlis.normalized_scrap_func(scrap_typ1_df,extend_list_to_drop=['tot'], tot_weight_col='tot' )
    # get the list of scrap
    scrap_list = aipi_utlis.get_features_list_dataset (scrap_typ1_norm_df, extend_list=['tot'])
   
    
    ### data input 3 data
    # preproccesing
    input3_typ1_df = utlis.preprocessing_func (input3_typ1_df, sort_col_list=['time','heat'])
    
    
    if y_element == 'element1':
       
        # create X and Y for modeling
        ## the dataset with suffix _heat means that there is a column with the name 'heat' which reresents the heat number
        X_heat, Y_heat, X, Y= aipi_utlis.create_X_Y_func (y_element, list_features= scrap_list,extend_list= ['heat'], 
                                                                                              list_dataset_model=[scrap_typ1_norm_df, analysis_typ1_df], test_size=0.0)
        
        best_model = find_ML_model_period (start_date, end_date, model_name='LinearRegression', data_root_dir_ML=path_save_model)
    
        
        if best_model is not None:
            path_save_model_priod = path_save_model + '\\'+ best_model
            loaded_data = pickle.load(open(path_save_model_priod, 'rb'))
            
            ###### some preprocessing before using saved model################
            
            #  It is possible that some features are missing in the test data set compared to the saved model.
            
            # get the features name in the saved model
            feature_names_model = loaded_data['model_metadata']['feature_names']
            # get the features name in the test dataset
            feature_test_data=X.columns
            # Identify missing features
            missing_features = list(set(feature_names_model).difference(feature_test_data))
            # for missing feature we add them to the test dataset with zero values
            if (len(missing_features)!=0):
                for feature in missing_features:
                    X[feature] = 0
            
            # It is possible that some features have been added in the test data set compared to the saved model
            # Identify extra features
            extra_features = list(set(feature_test_data).difference(feature_names_model))
            # drop the extra features from test dataset
            if (len(extra_features) !=0):
                X = X.drop(extra_features)
            
            # and finally we must have the features in the same order as the saved model
            X = X[feature_names_model]
            
            # use the saved model
            score_dics = utlis.apply_model (loaded_data['model'], X, Y)
            
                    
            # put the results of the ML in a dataframe 
            model_name =  loaded_data['model_metadata']['model_name']
            df_score = pd.DataFrame([{ model_name+'_r2':score_dics['r2'], model_name+'_MAE':score_dics['MAE'], model_name+'_RMSE':score_dics['RMSE'] }])


            metadata_modeling_df = pd.concat([metadata_modeling_df, df_score], axis=1).reset_index(drop=True)

    return metadata_modeling_df




## change the name w.r.t the file name
def input_file_csv_name (process_data_name_list):
    if process_data_name_list.startswith('data2_input'):
        input_name_api = 'data2'
    elif process_data_name_list.startswith('data3_input'):
        input_name_api = 'data3'
    
         
    return input_name_api


def prepare_json (metadata_df,tags_dict):
    dict_api_post = {}
    dict_api_post["measurement"] = tags_dict['stage']
    dict_api_post["fields"]= metadata_df.columns.to_list()
    dict_api_post["tags"] = list(tags_dict.keys())
    dict_api_post["values"] = [{"time": pd.to_datetime(metadata_df.loc[0,'Start_date']).strftime('%Y-%m-%d %H:%M:%S'),
                                "fields": metadata_df.iloc[0,:].to_list(), 
                               "tags": list(tags_dict.values())}]
    
    return dict_api_post



if not os.path.exists(data_root_dir_metadata):
    os.makedirs(data_root_dir_metadata)
    


# ## Transformation, Exploration and Modeling



# get the list of raw data dir 

raw_data_dir_list = [ name for name in os.listdir(data_root_dir_rawdata) if os.path.isdir(os.path.join(data_root_dir_rawdata, name)) ]
#check if the same dir exists for metadata 
for dir_name in raw_data_dir_list:
    path_metadata = data_root_dir_metadata+'\\'+dir_name
    if (os.path.exists(path_metadata)) and len(os.listdir(path_metadata)) != 0:
        print('metadata exists for', dir_name)
        continue
        
    else:
        print('\n create metadata for the raw data ', dir_name)
        
         # in order to get the typ2 and typ1 heats in each periodes we need the scrap_raw_data
        scrap_input_df = constants.TABLE_SCRAP_INPUT
        # give the file path
        pathfile = data_root_dir_rawdata+'\\'+dir_name+'\\'+scrap_input_df+'.csv'
        # read scrap data using class create_df
        scrap_class = utlis.create_df(pathfile)
        scrap_raw_df = scrap_class.read_data()
        # check heats for typ2 and typ1 steel
        heat_list_inconsistency, typ2_heats, typ1_heats = aipi_utlis.find_typ2_heat (scrap_raw_df)
        
        # Create data dir for metadata if not there yet
        #if not os.path.exists(data_root_dir_metadata+'\\'+dir_name):
        os.makedirs(data_root_dir_metadata+'\\'+dir_name)
        
        # a list of all process data in the dir 
        process_data_name_list = os.listdir(data_root_dir_rawdata+'\\'+dir_name)
        # for now drop temp and ppmoxygen
        process_data_name_list = [e for e in process_data_name_list if e not in ['ppmOxygen.csv','temp.csv']]


        # create Excel file to write transformation and exploration metadata
        # the path to save the metadata as excel file
        path_to_write = data_root_dir_metadata+'\\'+dir_name
        writer_transformtion =  pd.ExcelWriter(path_to_write+'\\'+'transformation.xlsx')
        writer_exploration =  pd.ExcelWriter(path_to_write+'\\'+'exploration.xlsx')
        writer_modeling =  pd.ExcelWriter(path_to_write+'\\'+'modeling.xlsx')
        
        
        ### create a dict to put all process data in it and use it to apply traied model 
        all_process_data_dict = {}
        
        # loop over process data for each periode
        for process_data_name in process_data_name_list:
                
            #### transformation
            

            # metadata transormation for process data
            process_metadata_transformation_df, process_typ1, start_date, end_date = metadata_extract_transformation (process_data_name,data_root_dir_rawdata, dir_name, typ1_heats,typ2_heats, scrap_raw_df)
            # replace Nan with '' to remove error in jason format
            Nan_rows = process_metadata_transformation_df[process_metadata_transformation_df.isnull().any(axis=1)]
            if (Nan_rows.shape[0] != 0):
                display(Nan_rows)

            process_metadata_transformation_df = process_metadata_transformation_df.fillna('')
            # write process data to excel table  
            process_metadata_transformation_df.to_excel(writer_transformtion, sheet_name=process_data_name.split('.')[0])
          
            # json format, the name of API input file can be found from input_file_csv_name func
            tags_dict_transformation={'input':input_file_csv_name (process_data_name),'stage':'transformation','period':raw_data_period}
            json_transformation = prepare_json (process_metadata_transformation_df,tags_dict=tags_dict_transformation)
            
            ################### post to the API
            print('post to API:', list(tags_dict_transformation.values()), json_transformation['values'][0]['time'])
            
            
            response_measurements_post = requests.post(SERVICE_ROOT_URL+'/measurements-set', headers=HEADERS, json=json_transformation)
            print('\n status_code:', response_measurements_post.status_code)
            print('\n response contents for transformation')
            print(response_measurements_post.content)
            print('---------')

            
            #### Exploration
     
            # metadata exploration for process data
            process_metadata_exploration_df = metadata_extract_exploration (process_data_name, process_typ1,start_date, end_date)
            # replace Nan with '' to remove error in jason format
            process_metadata_exploration_df = process_metadata_exploration_df.fillna('')
            # write process data to excel table  
            process_metadata_exploration_df.to_excel(writer_exploration, sheet_name=process_data_name.split('.')[0]) 
            
            # json format
            tags_dict_exploration={'input':input_file_csv_name (process_data_name), 'stage':'exploration','period':raw_data_period}
            json_exploration = prepare_json (process_metadata_exploration_df,tags_dict=tags_dict_exploration)
            
            ################### post to the API
            print('post to API:', list(tags_dict_exploration.values()), json_exploration['values'][0]['time'])
        
            response_measurements_post = requests.post(SERVICE_ROOT_URL+'/measurements-set', headers=HEADERS, json=json_exploration)
            print('\n status_code:', response_measurements_post.status_code)
            print('\n response contents exploration')
            print(response_measurements_post.content)
            print('---------')

            
           
            # save all process data to use them for modeling
            all_process_data_dict[process_data_name.split('.')[0]] = process_typ1
            
    
        metadata_modeling_df = metadata_extract_modeling(all_process_data_dict,start_date, end_date, y_element= 'Cu', path_save_model= data_root_dir_ML )
        # replace Nan with '' to remove error in jason format
        metadata_modeling_df = metadata_modeling_df.fillna('')
        metadata_modeling_df.to_excel(writer_modeling) 
        
        ################### post to the API
        # json format
        tags_dict_modeling={'stage':'modeling','period':raw_data_period}
        json_modeling = prepare_json (metadata_modeling_df,tags_dict=tags_dict_modeling)
        # post to the API
        print('post to API:', list(tags_dict_modeling.values()), json_modeling['values'][0]['time'])
        
        response_measurements_post = requests.post(SERVICE_ROOT_URL+'/measurements-set', headers=HEADERS, json=json_modeling)
        print('\n status_code:', response_measurements_post.status_code)
        print('\n response contents')
        print(response_measurements_post.content)
        print('---------')
           
        
        
    
        # Save workbook
        writer_exploration.close()
        writer_transformtion.close()
        writer_modeling.close()
    
    


# ## Trained models metadata



# get the list of all saved model
list_of_models = os.listdir(data_root_dir_ML) # this line gives all files in directory
model_list = [model for model in list_of_models if model.endswith('.sav')] # give only saved model with pickle

trained_model_df = pd.DataFrame() 

for count, model in enumerate(model_list, start=0):
    path_save_model_priod = data_root_dir_ML + '\\'+ model
    loaded_data = pickle.load(open(path_save_model_priod, 'rb'))
    
 
    
    model_metadata = loaded_data['model_metadata']
    
    ## start and end of the heat time
    trained_model_df.at[count,'Start_date'] = model_metadata['time_start']
    trained_model_df.at[count,'End_date'] = model_metadata['time_end']
    trained_model_df.at[count,'Nr_samples'] = model_metadata['Nr_samples']
    trained_model_df.at[count,'Nr_features'] = model_metadata['Nr_features']
    trained_model_df.at[count,'model_name'] = model_metadata['model_name']
    trained_model_df.at[count,'train_R2'] = model_metadata['train_R2']
    trained_model_df.at[count,'train_MAE'] = model_metadata['train_MAE']
    trained_model_df.at[count,'train_RMSE'] = model_metadata['train_RMSE']
    trained_model_df.at[count,'train_memory_used'] = model_metadata['memory_used']
    
    model_coeff = loaded_data['coef_model']
    for key in model_coeff.keys():
        trained_model_df.at[count,'coef_'+key] = model_coeff[key]
    
     # json format
    tags_dict_train_models={'stage':'train_models'}#,'period':raw_data_period}
    json_train_models = prepare_json (trained_model_df,tags_dict=tags_dict_train_models)
    
    # post to the API
    print('post to API:', list(tags_dict_train_models.values()), json_train_models['values'][0]['time'])
        
    response_measurements_post = requests.post(SERVICE_ROOT_URL+'/measurements-set', headers=HEADERS, json=json_train_models)
    print('\n status_code:', response_measurements_post.status_code)
    print('\n response contents train_models')
    print(response_measurements_post.content)
    print('---------')
            
    
writer_train_models =  pd.ExcelWriter(data_root_dir_train_model+'\\'+'train_models.xlsx')
trained_model_df.to_excel(writer_train_models) 
writer_train_models.close()


