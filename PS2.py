# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
from numpy.random import randint
import requests
import io
import timeit
from pprint import pprint

# # Question 0 - Code review warmup

# +
#sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]
#op = []

#for m in range(len(sample_list)):
#    li = [sample_list[m]]
#    for n in range(len(sample_list)):
#        if (sample_list[m][0] == sample_list[n][0] and sample_list[m][3] != sample_list[n][3]):
#            li.append(sample_list[n])
#    op.append(sorted(li, key=lambda dd: dd[3], reverse=True)[0])
#res = list(set(op))


# -

# The code takes a list of tuples and collects the ones with the same first and fourth element in a list. Finally it returns the tuple with the largest 4th element in the tuple among the ones with a common 1st element.
#
# The syntax  of the code can be improved as the number of comparisons it makes are much more than the necessary ones. Some smart indexing can avoid many of the comparison; farther the two nested loops are not necessary. Some comparisons are repated several times, which can be avoided by a previous sorting or ignoring comparisosn that were already made.
#
# The  style of the snippet is ok in some parts, but in other parts the variables' names do not provide significant meaning, as "li" or "op". What those variables are supposed to contain is not really clear. More meaningful and clear names could be used.
#
# The code does not really work because it compares the 4th element of the tuples, but the tuples only have 3 elements, therefore the comparison will not work in tuples of that size.
#
# Overall, the syntax of the code can definitely be improves, and even if the code works with larger tuples is very inefficient.
#

# # Question 1 - List of Tuples

# +
def tuple_generator(low=1, high=100, n=10, k=10):
    """
    Returns a list with n tuples with k randomly-generated integers 
    between low and high.
    Inputs:
        low: lower bound of random interval.
        high: upper bound of random interval.
        n: number of tuples, lenght of list.
        k: Number of random integers in each tuple.
    Output:
        List with n tuples with k integers each.
    """
    output = [tuple(np.sort(randint(low, high, size=k))) for i in range(n)]
    assert ((type(output)==list) and (type(output[0])==tuple)), 'Output with wrong format.'
    return output

tuple_list = tuple_generator(1, 10, 20, 5)


# -

# # Question 2

# +
def code_snippet(sample_list=tuple_list, ind_a=0, ind_b=2):
    """
    Function created with the code snippet.
    Inputs:
        smaple_list: List of tuples to sort and compare.
        ind_a: Index of 1st comaprison
        ind_b: Index of 2nd comparison
    Output:
        Sorted list of tuples.
    """
    out_list = []
    for m in range(len(sample_list)):
        li = [sample_list[m]]
        for n in range(len(sample_list)):
            if (sample_list[m][ind_a] == sample_list[n][ind_a] and sample_list[m][ind_b] != sample_list[n][ind_b]):
                li.append(sample_list[n])
        out_list.append(sorted(li, key=lambda dd: dd[ind_b], reverse=True)[0])
    output = list(set(out_list))
    return output 


def Improved_snippet(sample_list=tuple_list, ind_a=0, ind_b=2):
    """
    Improved version of the code snippet.
    Inputs:
        smaple_list: List of tuples to sort and compare.
        ind_a: Index of 1st comaprison
        ind_b: Index of 2nd comparison
    Output:
        Sorted list of tuples.
    """
    out_list = []
    sorted_list = sorted(sample_list, key=lambda x: x[ind_a])
    aux_var = sorted_list[0]
    for i in range(len(sorted_list)):
        if (aux_var[ind_a] != sorted_list[i][ind_a] or aux_var[ind_b] < sorted_list[i][ind_b]):
             aux_var = sorted_list[i]
        for j in range(i+1,len(sorted_list)):
            if (sorted_list[i][ind_a] == sorted_list[j][ind_a] and sorted_list[i][ind_b] < sorted_list[j][ind_b]):
                aux_var = sorted_list[j]
        out_list.append(aux_var)
    output = list(set(out_list))
    return output 


def Scratch_sort(sample_list=tuple_list, ind_a=0, ind_b=2):
    """
    Sorting algorithm from scratch.
    Inputs:
        smaple_list: List of tuples to sort and compare.
        ind_a: Index of 1st comaprison
        ind_b: Index of 2nd comparison
    Output:
        Sorted list of tuples.
    """
    output = []
    sorted_list = sorted(sample_list, key=lambda x: x[ind_a])
    aux_var = sorted_list[0]
    for i in range(1,len(sorted_list)):
        if aux_var[ind_a]!=sorted_list[i][ind_a]:
            output.append(aux_var)
            aux_var = sorted_list[i]
        else:
            if aux_var[ind_b]<sorted_list[i][ind_b]:
                aux_var = sorted_list[i]
    output.append(aux_var) 
    return output



# +
iters = 10000

t_code_snippet = timeit.repeat("code_snippet(tuple_list)",
                               setup="from __main__ import code_snippet, tuple_list",
                              repeat=10, number=iters)
t_code_improved = timeit.repeat("Improved_snippet(tuple_list)",
                                setup="from __main__ import Improved_snippet, tuple_list",
                               repeat=10, number=iters)
t_code_scratch = timeit.repeat("Scratch_sort(tuple_list)",
                               setup="from __main__ import Scratch_sort, tuple_list",
                              repeat=10, number=iters)

t_code_snippet  = np.array(t_code_snippet)/iters
t_code_improved  = np.array(t_code_improved)/iters
t_code_scratch  = np.array(t_code_scratch)/iters

str_t_snippet = "{:.1e}".format(t_code_snippet.mean())+" +- "\
                +"{:.1e}".format(t_code_snippet.std())+" s"
str_t_improved = "{:.1e}".format(t_code_improved.mean())+" +- "\
                +"{:.1e}".format(t_code_improved.std())+" s"
str_t_scratch = "{:.1e}".format(t_code_scratch.mean())+" +- " \
                +"{:.1e}".format(t_code_scratch.std())+" s"



# +
df = pd.DataFrame()
df["Time code_snippet"] = str_t_snippet,
df["Time improved_snippet"] = str_t_improved
df["Time scratch_sort"] = str_t_scratch


display(df)
#print(df.to_markdown())
# -

# # Question 3

# +
columns_demo = ["SEQN","RIDAGEYR","RIDRETH3","DMDEDUC2",
                "DMDMARTL","RIDSTATR","SDMVPSU","WTMEC2YR",
                "WTINT2YR"]

columns_demo_new = ["id", "age", "race", "education",
                    "marital status", "exam status", 
                   "pseudo-psu masked var", 
                   "2yr exam weight", "2yr interview weight"]

dtypes = [int, int, int, int, int, int, int, float, float]

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

pd.to_pickle(demo_data,'./Demo_data_2011_2018.p')


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

# -

print("Number of cases in Demographic data: "+str(len(demo_data))+".")
print("Number of cases in Oral data: "+str(len(oral_data))+".")



