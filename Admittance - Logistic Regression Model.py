#!/usr/bin/env python
# coding: utf-8

# ### Basics of Logistic Regression

# #### Import the relevant libraries

# In[2]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# #### Load the data

# In[4]:


raw_data = pd.read_csv('2.01. Admittance.csv')
raw_data


# In[5]:


data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
data


# #### Variables

# In[7]:


y = data['Admitted']
x1 = data['SAT']


# ### Let's plot the data

# #### Scatter Plot

# In[10]:


plt.scatter(x1,y,color='C0')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('Admitted', fontsize=20)
plt.show()


# ### Plot with a regression line

# In[14]:


x = sm.add_constant(x1)
reg_lin = sm.OLS(y,x)
results_lin = reg_lin.fit()

plt.scatter(x1,y,color = 'C0')
y_hat = x1 * results_lin.params[1]+results_lin.params[0]

plt.plot(x1, y_hat, lw=2.5, color='C8')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('Admitted', fontsize=20)
plt.show()


# In[15]:


# data is non-linear thus must use non-linear approach


# #### Plot with a Logistic Regression Curve

# In[22]:


reg_log = sm.Logit(y,x)
results_log = reg_log.fit()

def f(x,b0,b1):
    return np.array(np.exp(b0+x*b1) / (1 + np.exp(b0+x*b1)))

f_sorted = np.sort(f(x1,results_log.params[0], results_log.params[1]))
x_sorted = np.sort(np.array(x1))

plt.scatter(x1,y,color='C0')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('Admitted', fontsize=20)
plt.plot(x_sorted, f_sorted, color='C8')
plt.show()


# In[ ]:


# Logistic Regression Assumptions:
# 1. non-linear
# 2. no endogeneity
# 3. normality and homoscedasticity
# 4. no autocorrelation
# 5. no multicollinearity

# logistic Regression predicts the probability of an event occurring
# p(X) = exp(b0+b1*x1+...) / (1 + exp(b0+b1*x1+...))

# Logit Regression Model:
# p(X)/(1-p(X)) = exp(b0 + b1*x1 +...) 
# log(odds) = b0 + b1*x1 +...+ bk*xk

# Linear Regression Model:
# y = b0 + b1*x1 + ... + epsilon

