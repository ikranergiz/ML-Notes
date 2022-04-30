# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 12:01:23 2022

@author: Ikra
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

data = pd.read_csv("eksikveriler.csv")

# =================== Handling Missing Values ==========================

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

missing_values = data.iloc[:,3:4].values

# print(missing_values)

imputer = imputer.fit(missing_values)

data.iloc[:,3:4] = imputer.transform(missing_values)

# ========================= 1) Label Encoder ============================

le = preprocessing.LabelEncoder()

# Machine learning and transforming at the same time.
data.iloc[:,0] = le.fit_transform(data.iloc[:,0])

# =================== 2) One Hot Encoder ==================================

ohe = preprocessing.OneHotEncoder()

# .values returns np.ndarray like this -> [1 1 1 1 1 1 1 ]
# but one hot encoding wants (_ ,1) array so we need to convert it.
country_ndarray = data.iloc[:,0].values

# 'Series' object has no attribute 'reshape' but ndarray has it.
# -1 parameters means we don't know how many rows in the data.
# ndarray convert to the pandas series
country_one_col = country_ndarray.reshape((-1,1))

# We should give 1D array into the ohe.fit_transform.
data_ohe = ohe.fit_transform(country_one_col).toarray()

# ==================One Hot Encoding with Pandas ====================

print("The original data")
print(data)
print("*" * 30)

df_new = pd.get_dummies(data, columns=["ulke"], prefix="ulke")
print("The transform data using get_dummies")
print(df_new)

# ========================================================================
# NOTE: If you'd like to reshape pandas series, you need to
#       get values and then you can reshape these values. 
#     
# # import pandas library
# import pandas as pd
#   
# # make an array
# array = [2, 4, 6, 8, 10, 12]
#   
# # create a series
# series_obj = pd.Series(array)
#   
# # convert series object into array
# arr = series_obj.values
#   
# # reshaping series 
# reshaped_arr = arr.reshape((3, 2))
#   
# # show 
# reshaped_arr
# =============================================================================




























