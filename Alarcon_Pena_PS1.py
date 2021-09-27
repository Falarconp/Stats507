# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Question 0 - Markdown warmup

# This is *question 0* for problem set 1 of Stats 507.
#
# > Question 0 is about Markdown
#
# The next question is about the **Fibonacci sequence**, $F_n = F_{n-2} + F_{n-1}$. In part **a** we will define a Python function `fib_rec()`.
#
# Below is a ...
#
# ### Level 3 header
#
# Next, we can make a bulleted list:
#
# - Item 1
#     - detail 1
#     - detail 2
# - Item 2
#
# Finally we can make an enumerated list:
#
#     a. Item 1
#     b. Item 2
#     c. Item 3

# ```
#
# This is *question 0* for problem set 1 of Stats 507.
#
# > Question 0 is about Markdown
#
# The next question is about the **Fibonacci sequence**, $F_n = F_{n-2} + F_{n-1}$. In part **a** we will define a Python function `fib_rec()`.
#
# Below is a ...
#
# ### Level 3 header
#
# Next, we can make a bulleted list:
#
# - Item 1
#     - detail 1
#     - detail 2
# - Item 2
#
# Finally we can make an enumerated list:
#
#     a. Item 1
#     b. Item 2
#     c. Item 3
#   
# ```

# # Question 1 - Fibonnaci Sequence

# +
import numpy as np
import timeit
import pandas as pd

phi = (1 + np.sqrt(5))/2.   ## Golden number


def test_func(func):
    """
    Test function ensuring that
    the defined Fibonacci function returns F7,F11,F13 correctly.
    """
    F7 = func(7)
    F11 = func(11)
    F13 = func(13)
    if (F7==13) and (F11==89) and (F13==233):
        print("Function "+str(func)+" passed the test.")
    else:
        print ("Function "+str(func)+" did not pass the test.")
    return


def fib_rec(n, a=0, b=1):
    """
    fib_rec  function calculates the Fibonacci sequence recursively returning F_n.

    Inputs:
    n: # elements of the sequence
    a: F_1 (First element of the sequence, assumed to be 0 if not given)
    b: F_2 (Second  element of the sequence, assumed to be 1 if not given)

    """
    if type(n)!=int:
        print("n value given is not an integer. n should be a non-negative integer.")
        return
    if (n>1):
        return fib_rec(n-1)+fib_rec(n-2)
    elif n==0:
        return a
    elif n==1:
        return b
    else:
        print("n value given is not valid")
        return
    

test_func(fib_rec)
    

def fib_for(n, a=0, b=1):
    """
    fib_for  function calculates the Fibonacci sequence with a for loop returning F_n.

    Inputs:
    n: # elements of the sequence
    a: F_1 (First element of the sequence, assumed to be 0 if not given)
    b: F_2 (Second  element of the sequence, assumed to be 1 if not given)
    
    """
    if type(n)!=int:
        print("n value given is not an integer. n should be a non-negative integer.")
        return
    if n<0:
        print("n value given is not valid")
        return
    elif n==1:
        return b
    elif n==0:
        return a
    fn_2 = a
    fn_1 = b
    for i in range(n-1):
        Fn = fn_2+fn_1
        fn_2 = fn_1
        fn_1 = Fn
    return Fn

    
    
test_func(fib_for)


def fib_whl(n, a=0, b=1):
    """
    fib_whl  function calculates the Fibonacci sequence with a while loop returning F_n.

    Inputs:
    n: # elements of the sequence
    a: F_1 (First element of the sequence, assumed to be 0 if not given)
    b: F_2 (Second  element of the sequence, assumed to be 1 if not given)

    """
    if type(n)!=int:
        print("n value given is not an integer. n should be a non-negative integer.")
        return
    if n<0:
        print("n value given is not valid")
        return
    elif n==1:
        return b
    elif n==0:
        return a
    fn_2 = a
    fn_1 = b
    while (n>1):
        Fn = fn_2+fn_1
        fn_2 = fn_1
        fn_1 = Fn
        n-=1
    return Fn


test_func(fib_whl)


def fib_rnd(n, a=0, b=1):
    """
    fib_rnd  function calculates the Fibonacci sequence recursively the rounding method returning F_n.

    Inputs:
    n: # elements of the sequence
    a: F_1 (First element of the sequence, assumed to be 0 if not given)
    b: F_2 (Second  element of the sequence, assumed to be 1 if not given)

    """
    if type(n)!=int:
        print("n value given is not an integer. n should be a non-negative integer.")
        return
    if (n>1):
        return int(np.round(phi**n/np.sqrt(5)))
    elif n==0:
        return a
    elif n==1:
        return b
    else:
        print("n value given is not valid")
        return
    
    
test_func(fib_rnd)


def fib_flr(n, a=0, b=1):
    """
    fib_flr  function calculates the Fibonacci sequence using the truncation method returning F_n..

    Inputs:
    n: # elements of the sequence
    a: F_1 (First element of the sequence, assumed to be 0 if not given)
    b: F_2 (Second  element of the sequence, assumed to be 1 if not given)

    """
    if type(n)!=int:
        print("n value given is not an integer. n should be a non-negative integer.")
        return
    if (n>1):
        return int(phi**n/np.sqrt(5) + 0.5)
    elif n==0:
        return a
    elif n==1:
        return b
    else:
        print("n value given is not valid")
        return

    
test_func(fib_flr)



# +
setup = '''
import numpy as np

phi = (1 + np.sqrt(5))/2.   ## Golden number

def fib_rec(n, a=0, b=1):
    if type(n)!=int:
        print("n value given is not an integer. n should be a non-negative integer.")
        return
    if (n>1):
        return fib_rec(n-1)+fib_rec(n-2)
    elif n==0:
        return a
    elif n==1:
        return b
    else:
        print("n value given is not valid")
        return
    

def fib_for(n, a=0, b=1):
    if type(n)!=int:
        print("n value given is not an integer. n should be a non-negative integer.")
        return
    if n<0:
        print("n value given is not valid")
        return
    elif n==1:
        return b
    elif n==0:
        return a
    fn_2 = a
    fn_1 = b
    for i in range(n-1):
        Fn = fn_2+fn_1
        fn_2 = fn_1
        fn_1 = Fn
    return Fn

    
    



def fib_whl(n, a=0, b=1):
    if type(n)!=int:
        print("n value given is not an integer. n should be a non-negative integer.")
        return
    if n<0:
        print("n value given is not valid")
        return
    elif n==1:
        return b
    elif n==0:
        return a
    fn_2 = a
    fn_1 = b
    while (n>1):
        Fn = fn_2+fn_1
        fn_2 = fn_1
        fn_1 = Fn
        n-=1
    return Fn


def fib_rnd(n, a=0, b=1):
    if type(n)!=int:
        print("n value given is not an integer. n should be a non-negative integer.")
        return
    if (n>1):
        return int(np.round(phi**n/np.sqrt(5)))
    elif n==0:
        return a
    elif n==1:
        return b
    else:
        print("n value given is not valid")
        return
    
    



def fib_flr(n, a=0, b=1):
    if type(n)!=int:
        print("n value given is not an integer. n should be a non-negative integer.")
        return
    if (n>1):
        return int(phi**n/np.sqrt(5) + 0.5)
    elif n==0:
        return a
    elif n==1:
        return b
    else:
        print("n value given is not valid")
        return
        
    
        
'''

number_n = 11
iters = 10
n_repeat = 5
ns = np.linspace(0,30,number_n,dtype=int)
times = np.zeros([5,number_n])


for i in range(number_n):
    n = ns[i]
    times[0,i] = np.median(np.array(timeit.repeat("fib_rec("+str(n)+")",
                                 setup,repeat=10,number=iters)))/iters
    times[1,i] = np.median(np.array(timeit.repeat("fib_for("+str(n)+")",
                                 setup,repeat=10,number=iters)))/iters
    times[2,i] = np.median(np.array(timeit.repeat("fib_whl("+str(n)+")",
                                 setup,repeat=10,number=iters)))/iters
    times[3,i] = np.median(np.array(timeit.repeat("fib_rnd("+str(n)+")",
                                 setup,repeat=10,number=iters)))/iters
    times[4,i] = np.median(np.array(timeit.repeat("fib_flr("+str(n)+")",
                                 setup,repeat=10,number=iters)))/iters

    

# +
index_names = [str(i) for i in ns]
column_names = ["Time fib_rec()  [$\mu$sec]","Time fib_for()  [$\mu$sec]",
               "Time fib_whl()  [$\mu$sec]","Time fib_rnd()  [$\mu$sec]",
              "Time fib_flr()  [$\mu$sec]"]

df = pd.DataFrame(np.transpose(times*1e6), columns=column_names,index=index_names )
df.columns.name = 'n'
pd.set_option('display.precision', 2)
s = np.linspace(2,40,number_n,dtype=int)

display(df)
#print(df.to_markdown())
# -

# # Question 2 - Pascal's Triangle

# +
def Pascal_row_n(n):
    """
     Calculates the n-th row of Pascal's triangle
     The output is a np.array with the correspondin Binomial(n,k)
     """
    out = []
    out.append(1.0)
    for k in range(1,n+1):
        aux = out[-1]*(n+1-k)/k
        out.append(aux)
    return np.array(out,dtype=int)


def Print_Pascal_triangle(n,spacing=5):
    """
    Prints the n first n rows of Pascal's triangle.
    Inputs:
        n: # of rows.
        spacing: Dedicated space for each string. Default is 5.
     """
    #First iterates over each row starting from the top.
    for i in range(n+1):
        line = " "*spacing*(n-i+1)
        row_n = Pascal_row_n(i)
        # Iterates within the row itself to construct the string line.
        for k in range(i+1):
            aux = str(row_n[k])
            line += aux.ljust(spacing)
            #line += f"{row_n[k]:^5}"
            line += " "*spacing
        print(line+"\n")
    return


# -

Print_Pascal_triangle(10)

# # Question 3 - Statistics 101

# +
from scipy.stats import norm, binom, beta
import warnings

def compute_statistics(x,CI=0.95,
                       formt="{0:.3f} [{1:.0f}% CI: ({2:.2f} , {3:.2f})]"):
    """
    Calculates a point estimates and a confidence interval assuming a gaussian distribution.
    Inputs:
        x = 1-D array
        CI = Confidence interval
        formt = Format of output.
    Output
        String with point estimate and confidence interval or 
        Dictionary with keys "est","lwr","upr","level" if formt is None.
    """
    try:
        x = np.array(x,dtype=float)
    except:
        print("The function input is not a 1D array or 1D array-type")
        return
    std = x.std()
    mean = x.mean()
    z = (1+CI)/2.
    up = mean + norm.ppf(q=z)*std
    lw = mean + norm.ppf(q=1-z)*std
    line = formt.format(mean, 100*CI, lw, up)
    if formt==None:
        output = {}
        output["est"] = mean
        output["lwr"] = lw
        output["upr"] = up
        output["level"] = CI
        return output
    return line


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
    string = formt.format(n*p, 100*CI, 100 * lw, 100 * up)
    if formt==None:
        return output
    else:
        return string


# +
x = np.random.random(100)

print(compute_statistics(x, 0.1))

# +
array = np.concatenate([np.zeros(48),np.ones(42)])
np.random.shuffle(array)

CIs = [0.9,0.95,0.99]
meth_names = ["Normal","Clopper-Pearson","Jeffrey's","Agresti-Coull"]

df = pd.DataFrame(index=None,columns=meth_names)

for ci in CIs:
    lists = [None]*len(meth_names)
    lists[0] = compute_Bin(array,CI=ci,method="NA")
    lists[1] = compute_Bin(array,CI=ci,method="CP")
    lists[2] = compute_Bin(array,CI=ci,method="J")
    lists[3] = compute_Bin(array,CI=ci,method="AC")
    df.loc[len(df.index)] = lists
    


# -

display(df.style.hide_index())
#print(df.to_markdown(index=False))

df["Minimum width"] = ["Agresti-Coull","Normal/AgrestiCoull","Agresti-Coull"]
display(df.style.hide_index())
#print(df.to_markdown(index=False))


