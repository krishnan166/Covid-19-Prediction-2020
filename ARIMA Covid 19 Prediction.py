#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


no_confirmed = pd.read_csv("time_series_covid19_confirmed_global.csv")
no_deaths =  pd.read_csv("time_series_covid19_deaths_global.csv")
no_recovered =  pd.read_csv("time_series_covid19_recovered_global.csv")


# In[3]:


no_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True) 
no_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)
no_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)


# In[4]:


no_confirmed = no_confirmed.melt(id_vars=["Province/State","Country","Lat","Long"],var_name = "Date",value_name="Confirmed")
no_deaths = no_deaths.melt(id_vars=["Province/State","Country","Lat","Long"],var_name = "Date",value_name="Deaths")
no_recovered = no_recovered.melt(id_vars=["Province/State","Country","Lat","Long"],var_name = "Date",value_name="Recovered")


# In[5]:


no_confirmed["Deaths"] = no_deaths.Deaths
no_confirmed["Recovered"] = no_recovered.Recovered


# In[6]:


X = no_confirmed
X.Date = pd.to_datetime(X.Date)


# In[7]:


confirmed = X.groupby('Date').sum()['Confirmed']
deaths = X.groupby('Date').sum()['Deaths']
recovered = X.groupby('Date').sum()['Recovered']


# In[8]:


from pandas import read_csv
from matplotlib import pyplot
confirmed.hist()
pyplot.show()


# In[9]:


import pandas
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(confirmed)
pyplot.show()


# In[10]:


import statsmodels.api as sm
sm.graphics.tsa.plot_pacf(confirmed,lags='30')


# In[11]:


con = ARIMA(confirmed, order=(6,2,2))
fitting_con = con.fit(disp=0)
print(fitting_con.summary())


# In[12]:


fitting_con.plot_predict(dynamic=False)
plt.show()


# In[13]:


import pandas
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(deaths)
pyplot.show()


# In[14]:


death = ARIMA(deaths, order=(6,1,0))
fitting_death = death.fit(disp=0)
print(fitting_death.summary())


# In[15]:


fitting_death.plot_predict(dynamic=False)
plt.show()


# In[16]:


residuals = pd.DataFrame(fitting_con.resid)
residuals.plot(title="Residuals")
pyplot.show()


# In[17]:


residuals.plot(kind="kde",ylabel="Density")
pyplot.show()


# In[18]:


start_index = '2020-03-25'
end_index = '2021-07-10'
forecast = fitting_con.plot_predict(start=start_index, end=end_index)
#plt.show()


# In[19]:


arima_prediction = fitting_con.predict(dynamic=False)
arima_prediction


# In[20]:


y=confirmed[2:]
x=np.cumsum(arima_prediction.values)
print(x)


# In[21]:


mse = rmse(y,x)
print('RMSE: %f' % mse)


# In[22]:


MAE=metrics.mean_absolute_error(y,x)
print(f'Mean Absolute Error:{MAE}')


# In[23]:


EPSILON = 1e-10
def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted
def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    return _error(actual, predicted) / (actual + EPSILON)


# In[24]:


def rrse(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))

RRSE=rrse(y,x)
print(f'Root Relative Squared Error:{RRSE}')


# In[25]:


def mape(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.abs(_percentage_error(actual, predicted)))

MAPE=mape(y,x)
print(f'Mean Absolute Percentage Error:{MAPE}')


# In[ ]:




