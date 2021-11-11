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
#     display_name: local-venv
#     language: python
#     name: local-venv
# ---

# # Sparse Data Structures - falarcon@umich.edu

# `pandas` offers a way to speed up not only calculations in the typical `sparse` meaning, i.e. , `DataFrames` with 0's, but also for particular values or `NaN's`.
#
#
# Let's first start showing the effect it has on discarding `NaN's` or a particular values and compare it with other methods. 
#
# The goal of using `sparse` Data Structures is to allocate memory efficiently in large data sets and also speed-up possible operations between `sparse` Data Structures. `Sparse Data Structure` saved values and locations instead of the whole Dataframe. 
#

import pandas as pd
import numpy as np
import tabulate

# +
## Creating toy data.

array_nans = np.random.randn(500, 10000)

# Switching some values to NaN's to produce a sparse structure.
array_nans[10:499,1:9900] = np.nan

dense_df = pd.DataFrame(array_nans)
sparse_df = dense_df.astype(pd.SparseDtype("float", np.nan))

print(" Density of sparse DataFrame: "+str(sparse_df.sparse.density))
# -

# ## Efficiency in storing Sparse DataStructures
#
# `Sparse DataStructures` are more efficient in allocating memory for large datasets with lots of NaN's or information that it is not of interest. The toy data has some sparsity $\sim$ 50%, but real data or matrices could have densities of the orden of $\sim$0.1 % or less.

# +
## Let's compare the storing times for different methods and the same datastructure  being sparse or not.

print('Dense data structure : {:0.3f} bytes'.format(dense_df.memory_usage().sum() / 1e3))
print('Sparse data structure : {:0.3f} bytes'.format(sparse_df.memory_usage().sum() / 1e3))
# -

# Even though the sparse allocate memory better, thy take slightly longer to be created. Nevertheless, we will prove that when there are heavy operations being undertaken in large sparse data structures, the speed-up is worth it, and the allocation of memory as well.

# %timeit  df_2 = pd.DataFrame(array_nans)

# %timeit  sparse_df = pd.DataFrame(array_nans).astype(pd.SparseDtype("float", np.nan))

# ## Speed-up of calculations in Sparse DataStructures and comparison with scipy.
#
# Finally we compare the time it takes to operate on `Dense DataStructures` and `Sparse DataStructures`. Operating directly on `Sparse DataStructures` is not really efficient because `pandas` converts them to `Dense DataStructures` first. Nevertheless the `scipy` package has methods that take advantage of the psarsity of matrices to speed-up those processes.

# +
## scipy also offers methods for sparse arrays, although in the full with 0's meaning,
from scipy.sparse import csr_matrix

rnd_array = np.zeros((10000,500))
rnd_array[200:223,13:26] = np.random.randn(23,13)
sparse_rnd_array = csr_matrix(rnd_array)
sparse_matrix_df = csr_matrix(sparse_df)
# -

# %timeit sparse_matrix_df.dot(sparse_rnd_array)

# %timeit dense_df.dot(rnd_array)

# As we can see `Sparse` methods are specially useful to manage data with repeated values or just values we are not interested in. It can also be used to operate on them using `scipy` and its methods for `sparse` arrays, which could be much faster that standard multiplications. It is important to notice that it is only faster when the sparsity is significant, usually less than 1%.
