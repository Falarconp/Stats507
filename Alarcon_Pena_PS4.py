# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt


# # Question 0 - Topics in Pandas
#
# ## Sparse Data Structures
#
#
# `pandas` offers a way to speed up not only calculations in the typical `sparse` meaning, i.e. , `DataFrames` with 0's, but also for particular values or `NaN's`.
#
#
# Let's first start showing the effect it has on discarding `NaN's` or a particular values and compare it with other methods. 
#

# +
## Creating toy data with several NaN's


## Creating toy data with several NaN's
# -

# ## Speed-up in storing Sparse DataStructures
#
# Not only that but `Sparse DataStructures` are faster at storing data of interest, which is of particular interest when we want to discard some values or are only interested on a subset of the data.

# +
## Let's compare the storing times for different methods and the same datastructure  being sparse or not.
# -

# ## Speed-up of calculations in Sparse DataStructures and comparison with scipy.
#
# Finally we compare the time it takes to operate on `Sparse DataStructure` comparing with time period it takes for normal operations and similar methos in the `scipy` library.

# +
## scipy also offers methods for sparse arrays, although in the full with 0's meaning,
## while pandas is more complete.
# -

# As we can see `Sparse` methods are specially useful to manage data with repeated values or just values we are not interested in. It can also be used to operate on them at a similar speed than `scipy` methods, both of them much faster that standar multiplications.

# # Question 1 - NHANES Table 1
#
# ### part a) 

# +
columns_demo = ["SEQN","RIDAGEYR","RIDRETH3","DMDEDUC2",
                "DMDMARTL","RIDSTATR","SDMVPSU","WTMEC2YR",
                "WTINT2YR","RIAGENDR"]

columns_demo_new = ["id", "age", "race", "education",
                    "marital_status", "exam_status", 
                   "pseudo-psu_masked_var", 
                   "2yr_exam_weight", "2yr_interview_weight", "gender"]

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
#demo_data.dropna(inplace=True)
#Change Column Names.
demo_data.rename(columns=rename_cols, inplace=True)
#Change data types.
#demo_data = demo_data.astype(data_types)
# -

# ### part b)

# +
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
columns_oral_new = ["id", "ohx_status"]
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


# +
df = pd.merge(demo_data[["id","gender","age","education", "exam_status"]],
         oral_data[["id","ohx_status"]], on="id")

df["under_20"] = df.age<20
df["under_20"] = df["under_20"].astype("category")

df["college"] = pd.Categorical(np.where(df.education > 3,
                                        "some college/college graduate","No college/<20"))

df["ohx"] = pd.Categorical(np.where((df["exam_status"]==2) & (df["ohx_status"]==1),
                                     "complete","partial/incomplete"))


#index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])


#df.groupby("under_20" ,as_index=True).agg(set)

#df.groupby("under_20" ,as_index=True).agg(set).reset_index()
# -

#df.columns.to_flat_index()
df.groupby("under_20" ,as_index=True).agg(to_flat_index)


pd.MultiIndex.from_arrays(df)

df.groups

# ### part c)

df.drop(df[df['exam_status']!=2].index, inplace=True) 

# ### part d)

# # Question 2 - Monte Carlo Comparison 
#
# ### a) Level Calibration

# +
from numpy.random import default_rng
from copy import copy

rng = default_rng(seed=12)
p_s = rng.random(30)
p_s = np.sort(p_s[p_s<0.5])

number_ps = len(p_s)

N_samples = 50000
n_s = default_rng().random(N_samples)

grid_values = np.zeros((number_ps,N_samples))

for i in range(number_ps):
    aux = copy(n_s)
    aux[n_s<p_s[i]] = 1
    aux[n_s>p_s[i]] = 0
    grid_values[i,:] = aux
    
p_est = np.mean(grid_values, axis=-1)


# +
def compute_Bin(array,CI=0.95, method="NA",
                formt="{0:.1f}% [{1:.0f}% CI: ({2:.1f} , {3:.1f})%]"):
    """
    Calculates confidence interval for a population from a binomial experiment.
    Inputs:
        array: 1D numpy.array or 1D numpy.array like object.
        method: {"NA","CP","J","AC"}. Default is NA.
        CI: Confidence interval, default is 0.95
        formt = Format of output.
    Output:
        String with point estimate and confidence interval or 
        Dictionary with keys "est","lwr","upr","level" if formt is None.
    """
    from scipy.stats import norm, binom, beta
    import warnings
    try:
        array = np.array(array,dtype=float)
    except:
        print("The function input is not a 1D array or 1D array-type")
        return
    output = {}
    p = sum(array)/array.size
    n = array.size
    x = np.sum(array)
    alpha = (1-CI)
    z = (1+CI)/2.
    if method=="NA":
        std = np.sqrt(p*(1-p)/n)
        lw = p + norm.ppf(q=alpha)*std
        up = p + norm.ppf(q=1-alpha)*std
        output["est"] = n*p
        output["lwr"] = n*lw
        output["upr"] = n*up
        output["level"] = CI
        if (p*n>12) or ((1-p)*n>12):
           warnings.warn("The approximation may not be adequate.") 
    elif method=="CP":   #Clopper-Pearson Method
        lw = beta.ppf(alpha/2.,x,n-x+1)
        up = beta.ppf(1-alpha/2.,x+1,n-x)
        output["est"] = n*p
        output["lwr"] = n*lw
        output["upr"] = n*up
        output["level"] = CI
    elif method=="J":   #Jeffrey's Method
        lw = beta.ppf(alpha/2,x+0.5,n-x+0.5)
        up = beta.ppf(1-alpha/2.,x+1,n-x+0.5)
        output["est"] = n*p
        output["lwr"] = n*lw
        output["upr"] = n*up
        output["level"] = CI
    elif method=="AC":   #Agresti-Coull
        n_AC = n+z**2
        p_AC = (x+z**2/2.)/n_AC
        std = np.sqrt(p_AC*(1-p_AC)/n)
        lw = p_AC + norm.ppf(q=alpha)*std
        up = p_AC + norm.ppf(q=1-alpha)*std
        output["est"] = n*p_AC
        output["lwr"] = n*lw
        output["upr"] = n*up
        output["level"] = CI 
    else:
        print("Method is not valid. Use NA, CP, J or AC.")
        return
    string = formt.format(p, 100*CI, lw, up)
    if formt==None:
        return output
    else:
        return string



# +
CI_s = [0.8,0.9,0.95]
output_estimates = []

for ci in CI_s:
    results = [compute_Bin(grid_values[i,:],CI=ci,method="J",formt="{0:.3f} [{1:.0f}% CI: ({2:.3f} , {3:.3f})]") for i in range(number_ps)]
    output_estimates.append(results)
# -

# ### b) Relative Efficiency




