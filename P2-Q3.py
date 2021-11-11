# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
from numpy.random import randint
import requests
import io
import timeit

columns_demo = ["SEQN","RIDAGEYR","RIDRETH3","DMDEDUC2",
                "DMDMARTL","RIDSTATR","SDMVPSU","WTMEC2YR",
                "WTINT2YR","RIAGENDR"]

columns_demo_new = ["id", "age", "race", "education",
                    "marital status", "exam status", 
                   "pseudo-psu masked var", 
                   "2yr exam weight", "2yr interview weight", "gender"]

dtypes = [int, int, int, int, int, int, int, float, float, int]

rename_cols = dict(zip(columns_demo, columns_demo_new))
data_types = dict(zip(columns_demo_new, dtypes))

yr_11_12 = pd.read_sas('./DEMO_G.XPT')[columns_demo]
yr_11_12["period"] =  "2011-2012"
yr_13_14 = pd.read_sas('./DEMO_H.XPT')[columns_demo]
yr_13_14["period"] =  "2013-2014"
yr_15_16 = pd.read_sas('./DEMO_I.XPT')[columns_demo]
yr_15_16["period"] =  "2015-2016"
yr_17_18 = pd.read_sas('./DEMO_J.XPT')[columns_demo]
yr_17_18["period"] =  "2017-2018"

# Stack all the different dataframes into one.
demo_data = pd.concat([yr_11_12, yr_13_14,
                       yr_15_16, yr_17_18],
                      ignore_index=True)

#Drop NaNs.
demo_data.dropna(inplace=True)
#Change Column Names.
demo_data.rename(columns=rename_cols, inplace=True)
#Change data types.
demo_data = demo_data.astype(data_types)
demo_data["gender"] = pd.Categorical(np.where(demo_data.gender > 1.0,"female","male"))

pd.to_pickle(oral_data,'./Demo_data_2011_2018.p')

columns_oral = ["SEQN","OHDDESTS"]


#Creating mask for the required columns
or_11_12 = pd.read_sas('./OHXDEN_G.XPT')
mask1 = or_11_12.columns.str.contains("TC")
mask2 = ((or_11_12.columns=="SEQN") | (or_11_12.columns=="OHDDESTS"))
mask3 = or_11_12.columns.str.contains("CTC")
mask = mask1 | mask2
columns_oral = or_11_12.columns[mask]

or_11_12[columns_oral]
or_11_12["period"] =  "2011-2012"
or_13_14 = pd.read_sas('./OHXDEN_H.XPT')[columns_oral]
or_13_14["period"] =  "2013-2014"
or_15_16 = pd.read_sas('./OHXDEN_I.XPT')[columns_oral]
or_15_16["period"] =  "2015-2016"
or_17_18 = pd.read_sas('./OHXDEN_J.XPT')[columns_oral]
or_17_18["period"] =  "2017-2018"


dtypes_oral = [int]*2+[int]*sum(mask3 & mask1)+[str]*sum(mask3)
columns_oral_new = ["seq oral number", "Dentition Status"]
columns_oral_new += [str(i)+" tooth count" for i in range(sum(mask3 & mask1))]
columns_oral_new += ["coronal caries "+str(i)+" tooth" for i in range(sum(mask3))]
rename_cols_oral = dict(zip(columns_oral, columns_oral_new))
data_types_oral = dict(zip(columns_oral_new, dtypes_oral))

#Stacking the data
oral_data = pd.concat([or_11_12[columns_oral], or_13_14, 
                       or_15_16, or_17_18], ignore_index=True)
#Drop NaN's.
oral_data.dropna(inplace=True)
#Change column names.
oral_data.rename(columns=rename_cols_oral, inplace=True)
#Change datatypes.
oral_data = oral_data.astype(data_types_oral)

#Save data in a pickle
pd.to_pickle(oral_data,'./Oral_data_2011_2018.p')

