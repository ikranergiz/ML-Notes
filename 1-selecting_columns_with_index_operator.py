# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:24:08 2022

@author: Ikra Nergiz

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# ==============================================================

#Relative Path
veriler = pd.read_csv('eksikveriler.csv') #Comma Seperated Value

#Absolute Path -> Full Path

# ==============================================================

boy = veriler['boy']

# Returns pandas Series.
print(type(boy))

"""
    It’s possible to select multiple columns with just the indexing operator 
    by passing it a list of column names. Let’s select boy and kilo: 
"""

boy = veriler[['boy','kilo']]

# Returns pandas DataFrame.
print(type(boy))

"""
    You can actually select a single column as a DataFrame with a one-item list:

"""

# Returns DataFrame.
print(type(veriler[['boy']]))

# ==============================================================

#Missing Values Handling with sci-kit learn

from sklearn.impute import SimpleImputer

# Strategy decides what value to put in NaN.
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

# .values returns data without any labels such as index and column name.
yas = veriler.iloc[:,1:4].values 

print(yas)

# This is for learning. It learns that mean of data.
imputer = imputer.fit(yas[:,0:3])

# This is for changing values.
yas[:, 0:3] = imputer.transform(yas[:,0:3])

print(yas)

# ==============================================================

# Examples about Using loc and iloc

# Returns 7th row.
print(veriler.loc[7])


#If you want to choose much more row, you should use [[]].
print(veriler.loc[[7,3,5]])


# You can use slice notation to select a range of rows.
print(veriler.loc[7:13])


# If you want to walk 2 instead of 1.
print(veriler.loc[7:13:2])


"""
    You can choose rows and columns at the same time
    print(veriler.loc[row_selection, column_selection])

"""

# Choosing rows and columns using list.
print(veriler.loc[[2,5,6], ['boy', 'kilo']])

# Notice that we don't need to use [] while sciling.
print(veriler.loc[3:10, ['kilo']])

# Select a single row and single column.
# It returns that 'kilo' of 7th row.
print(veriler.loc[7, 'kilo'])


#==============================================================
