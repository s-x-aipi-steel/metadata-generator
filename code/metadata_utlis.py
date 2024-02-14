#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

# Self-made modules
#import utlis
import code.utlis as utlis


# ## Metadata Transformation 
#    - Metadata for Transformation which includes:
#    - Number of heats 
#    - Number of zeros or Nans
#    - Statistical features



class metadata_transformation_class:
    '''
    in this class the metadata for step transformation is calculated
        -  get the number of heats for typ1 and typ2
        -  Number of zeros in the typ1 heats
        -  Number of Nan values
        -  Statistical properties of each elements
    '''
    def __init__ (self, data_df, start_date, end_date, typ1_heats=list[float], typ2_heats=list[float]):
        self.data_df = data_df
        # get the typ1 heats
        self.data_typ1_df = self.data_df[self.data_df['ncol'].isin(typ1_heats)].reset_index(drop=True)
        # get the typ2 heats
        self.data_typ2_df = self.data_df[self.data_df['ncol'].isin(typ2_heats)].reset_index(drop=True)
        self.start_date = start_date
        self.end_date = end_date
        
    def return_typ1_df (self):
        return self.data_typ1_df
    
    def return_total_df (self):
        return self.data_df
        
    def add_extra_features (self, list_new_features=list[dict]):
        # list_new_features is a list which is in the format of [{new_name: [sum list of columns]},...]
       
        # loop over a list which contains the new features
        for i in range(len(list_new_features)):
            # for the new feaure get the new name and the name of columns to sum up
            for new_col_name, col_list in list(list_new_features[i].items()):
                self.data_df[new_col_name] = self.data_df[col_list].sum(axis=1)
                
        return self.data_df
        

    def Metadata_heats_info (self):
        # create a dataframe to put all infos about heats
        heat_info_df = pd.DataFrame() 
        
        ## start and end of the heat time
        
        heat_info_df.at[0,'Start_date'] = self.start_date
        heat_info_df.at[0,'End_date'] = self.end_date
        
        # get the number of heats
        heat_info_df.at[0,'Nr_heats'] = int(len(self.data_df['ncol']))
        
        return heat_info_df
    
    def Metadata_statistical (self, columns: list[str]):
        
        ### Creating a data framework for the statistical description  
        describe_df = pd.DataFrame()
        
        ## start and end of the heat time, later use as key merge
        describe_df.at[0,'Start_date'] = self.start_date
        describe_df.at[0,'End_date'] = self.end_date
        
        # loop over selected columns of dataset
        for column_name in columns:
            column = self.data_df[column_name]
            
            # add statistical description to the df
            describe_df.at[0,column_name+str('_min')]= float(column.min())
            describe_df.at[0,column_name+str('_mean')]= float(column.mean())
            describe_df.at[0,column_name+str('_std')]= float(column.std())
            describe_df.at[0,column_name+str('_max')]= float(column.max())

        return describe_df

    def Metadata_number_zeros (self, columns: list[str]):
        ### Creating a data framework for zeros
        zeros_df = pd.DataFrame()
        
        ## start and end of the heat time, later use as key merge
       
    
        zeros_df.at[0,'Start_date'] = self.start_date
        zeros_df.at[0,'End_date'] = self.end_date
        
        #  Count number of zeros in selected columns of Dataframe
        for column_name in columns:
            column = self.data_df[column_name]
            # Get the count of Zeros in column 
            count = (column == 0).sum()
            
            zeros_df.at[0,column_name+str('_zeros')]=int(count)
        
        return zeros_df
    

    
    def Metadata_number_Nans (self, columns: list[str]):
        ### in the case of data3_input number of zeros are Nans
        ### Creating a data framework for Nans
        Nans_df = pd.DataFrame()
        
        ## start and end of the heat time, later use as key merge
            
        Nans_df.at[0,'Start_date'] = self.start_date
        Nans_df.at[0,'End_date'] = self.end_date
        
        #  Count number of zeros in selected columns of Dataframe
        for column_name in columns:
            column = self.data_df[column_name]
            # Get the count of Nans in column 
            count = (column == 0).sum()
            
            Nans_df.at[0,column_name+str('_Nan')]=int(count)
        
        return Nans_df
        

    

    


# ## Metadata exploration
#    - Upper an lower limits of the variables
#    - number of outliears
# 
# __the dataset should be original or after data preprocessing?__
#     for example removing zeros and rows with missing values




class metadata_exploration_class:
    
     def __init__ (self, data_typ1_df, start_date, end_date):
        ## data_typ1_df is the data set before preprocessing
        self.data_typ1_df = data_typ1_df
        self.start_date = start_date
        self.end_date = end_date
    
     def Metadata_heats_info (self):
        # create a dataframe to put all infos about heats
        heat_info_df = pd.DataFrame() 
        
        ## start and end of the heat time
      
        heat_info_df.at[0,'Start_date'] = self.start_date
        heat_info_df.at[0,'End_date'] = self.end_date
        
        # get the number of heats
        heat_info_df.at[0,'Nr_heats'] = int(len(self.data_typ1_df['ncol']))
        
        return heat_info_df
    
     def Metadata_outliers(self, columns_dir, method='std', nstd=4, niqr=1.5):
        # a dir of columns name it also contains mention consider upper and lower limits flags to get the outliers.
        # d = {'col1': {'upper': True, 'lower': False}, 'col2': {'upper': True, 'lower': Treu}, ....}

        # lower the values lower than lower limits are outliers
        # upper if the values higher than upper limit are outliers
        # method can be 'std' or 'iqr'
        # if std is chosen the nstd can select
        # if iqr is chosen the niqr can select
        
        # the keys of the dir as a list
             
        # create a df to put outliers metadata
        outliers_infos_df = pd.DataFrame()

        ## start and end of the heat time, later use as key merge
        
        outliers_infos_df.at[0,'Start_date'] = self.start_date
        outliers_infos_df.at[0,'End_date'] = self.end_date
        
       
    
        columns = list(columns_dir.keys())
        if len(columns)!=0:
            data_oulier = self.data_typ1_df[columns]
            # there  is a class in utlis names outliers
            outlier_class = utlis.outliers(data_oulier)
            dics_limits_std, dics_limits_iqr = outlier_class.find_limits_outlier (col_list= data_oulier.columns, nstd=nstd, k=niqr)
            if method == 'std' :
                dics_input_limits = dics_limits_std
            elif method == 'iqr' :
                dics_input_limits = dics_limits_iqr

        

            ### the limit is a dict which includes the columns name with upper and lower boundries
            for item in dics_input_limits.items():
                # if lower is true means the values lower than lower limit are outliers
                # if lower is False means the lower values are anomalies and we set the lower limit to nan
                # the flags for upper and lower from column_dir input
                lower = columns_dir[ item[0]]['lower']
                upper = columns_dir[ item[0]]['upper']
                if lower:
                    outliers_infos_df.at[0,str('lower_boundary_')+item[0]] =float(item[1][0])
                else:
                    item[1][0] = np.nan

                # if Upper is true means the values higher than upper limit are outliers
                # if upper is False means the higher values are anomalies and we set higher limit to nan
                if upper:
                    outliers_infos_df.at[0,str('upper_boundary_')+item[0]] =float(item[1][1])
                else:
                    item[1][1] = np.nan


                # to get the outliers one by one
                # create a dict
                item_dic = {}
                # put the upper and lower for each column
                item_dic [item[0]]=[item[1][0],item[1][1]]
                # remove the elements below the lower limit and higher than the upper limit
                remove_ouliers_data_typ1_df = utlis.drop_outliers_rows_col_seperate (self.data_typ1_df, item_dic)

                # count the number of outliers with subtracting the number of heat before and after removing the outliers
                Nr_outliers =  self.data_typ1_df.shape[0]- remove_ouliers_data_typ1_df.shape[0]
                outliers_infos_df.at[0,str('Nr_outliers_')+item[0]] = Nr_outliers
                # relative Nr of outliers to Nr heats for this period
                outliers_infos_df.at[0,str('percent_Nr_outliers_Nr_heats_')+item[0]] = 100*Nr_outliers/(self.data_typ1_df.shape[0])

          
            
         
        return outliers_infos_df
    
    
     def Metadata_anomalies(self, columns_dir, method='std', nstd=4, niqr=4):
         # a dir of columns name it also contains mention consider upper and lower limits flags to get the outliers.
        # d = {'col1': {'upper': True, 'lower': False}, 'col2': {'upper': True, 'lower': Treu}, ....}
        
        # lower the values lower than lower limits are anomalies
        # upper if the values higher than upper limit are anomalies
        # method can be 'std' or 'iqr'
        # if std is chosen the nstd can select
        # if iqr is chosen the niqr can select
        
        
            
        # create a df to put outliers metadata
        anomalies_infos_df = pd.DataFrame()

        ## start and end of the heat time, later use as key merge
     
        anomalies_infos_df.at[0,'Start_date'] = self.start_date
        anomalies_infos_df.at[0,'End_date'] = self.end_date

        
        # the keys of the dir as a list
       
        columns = list(columns_dir.keys())
        
        if len(columns) !=0 :
            data_anomaly = self.data_typ1_df[columns]
            # there  is a class in utlis names outliers
            outlier_class = utlis.outliers(data_anomaly)
            dics_limits_std, dics_limits_iqr = outlier_class.find_limits_outlier (col_list= data_anomaly.columns, nstd=nstd, k=niqr)
            if method == 'std' :
                dics_input_limits = dics_limits_std
            elif method == 'iqr' :
                dics_input_limits = dics_limits_iqr



        

            ### the limit is a dict which includes the columns name with upper and lower boundries
            for item in dics_input_limits.items():
                # if lower is true means the values lower than lower limit are anomalies
                # if lower is False means the lower values are anomalies and we set the lower limit to nan
                # the flags for upper and lower from column_dir input
                lower = columns_dir[ item[0]]['lower']
                upper = columns_dir[ item[0]]['upper']
                if lower:
                    anomalies_infos_df.at[0,str('lower_boundary_')+item[0]] =float(item[1][0])
                else:
                    item[1][0] = np.nan

                # if Upper is true means the values higher than upper limit are anomalies
                # if upper is False means the higher values are anomalies and we set higher limit to nan
                if upper:
                    anomalies_infos_df.at[0,str('upper_boundary_')+item[0]] =float(item[1][1])
                else:
                    item[1][1] = np.nan
            
          
                # to get the outliers one by one
                # create a dict
                item_dic = {}
                # put the upper and lower for each column
                item_dic [item[0]]=[item[1][0],item[1][1]]
                # remove the elements below the lower limit and higher than the upper limit
                remove_anomalies_data_typ1_df = utlis.drop_outliers_rows_col_seperate (self.data_typ1_df, item_dic)

                # count the number of outliers with subtracting the number of heat before and after removing the outliers
                Nr_anomalies =  self.data_typ1_df.shape[0]- remove_anomalies_data_typ1_df.shape[0]
                anomalies_infos_df.at[0,str('Nr_anomalies_')+item[0]] = Nr_anomalies
                # relative Nr of outliers to Nr heats for this period
                anomalies_infos_df.at[0,str('percent_Nr_anomalies_Nr_heats_')+item[0]] =100* Nr_anomalies/(self.data_typ1_df.shape[0])

          
                 
        return anomalies_infos_df

