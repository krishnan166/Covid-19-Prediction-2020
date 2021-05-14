#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn import metrics
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[50]:


no_confirmed = pd.read_csv("time_series_covid19_confirmed_global.csv")
no_deaths =  pd.read_csv("time_series_covid19_deaths_global.csv")
no_recovered =  pd.read_csv("time_series_covid19_recovered_global.csv")
no_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True) 
no_recovered.rename(columns={'Country/Region':'Country'}, inplace=True)
no_deaths.rename(columns={'Country/Region':'Country'}, inplace=True)
no_confirmed = no_confirmed.melt(id_vars=["Province/State","Country","Lat","Long"],var_name = "Date",value_name="Confirmed")
no_deaths = no_deaths.melt(id_vars=["Province/State","Country","Lat","Long"],var_name = "Date",value_name="Deaths")
no_recovered = no_recovered.melt(id_vars=["Province/State","Country","Lat","Long"],var_name = "Date",value_name="Recovered")
no_confirmed["Deaths"] = no_deaths.Deaths
no_confirmed["Recovered"] = no_recovered.Recovered


# In[51]:


X = no_confirmed
confirmed = X.groupby('Date').sum()['Confirmed'].reset_index()
deaths = X.groupby('Date').sum()['Deaths'].reset_index()
recovered = X.groupby('Date').sum()['Recovered'].reset_index()
confirmed.columns = ['ds','y']
confirmed['ds'] = pd.to_datetime(confirmed['ds'])


# In[52]:


m = Prophet(interval_width=0.95,yearly_seasonality=True,daily_seasonality=True)
m.fit(confirmed)


# In[53]:


future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)
forecast


# In[54]:


confirmed_forecast_plot = m.plot(forecast)


# In[55]:


deaths.columns = ['ds','y']
deaths['ds'] = pd.to_datetime(deaths['ds'])
m_deaths = Prophet(interval_width=0.95,yearly_seasonality=True,daily_seasonality=True)
m_deaths.fit(deaths)
future_deaths = m_deaths.make_future_dataframe(periods=365)
forecast_deaths = m_deaths.predict(future_deaths)
forecast_deaths


# In[56]:


confirmed_forecast_plot_deaths = m_deaths.plot(forecast_deaths)


# In[58]:


recovered.columns = ['ds','y']
recovered['ds'] = pd.to_datetime(recovered['ds'])
m_recovered = Prophet(interval_width=0.95,yearly_seasonality=True,daily_seasonality=True)
m_recovered.fit(recovered)
future_recovered = m_recovered.make_future_dataframe(periods=7)
forecast_recovered = m_recovered.predict(future_recovered)


# In[59]:


prophet_pred = pd.DataFrame({"Date" : forecast[:63]['ds'], "Pred" : forecast[:63]["yhat"]})
prophet_pred = prophet_pred.set_index("Date")
prophet_pred


# In[60]:


confirmed.values[:,1]


# In[61]:


prophet_pred.Pred


# In[62]:


prophet_rmse_error = rmse(confirmed.values[:,1],prophet_pred.Pred)
print(f'RMSE Error: {prophet_rmse_error}')


# In[63]:


MAE=metrics.mean_absolute_error(confirmed.values[:,1],prophet_pred.Pred )
print(f'Mean Absolute Error:{MAE}')


# In[64]:


EPSILON = 1e-10
def _error(actual: np.ndarray, predicted: np.ndarray):
    return actual - predicted

def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    return _error(actual, predicted) / (actual + EPSILON)


# In[65]:


def rrse(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))
RRSE=rrse(confirmed.values[:,1],prophet_pred.Pred)
print(f'Root Relative Squared Error:{RRSE}')


# In[44]:


def mape(actual: np.ndarray, predicted: np.ndarray):
    return np.mean(np.abs(_percentage_error(actual, predicted)))
MAPE=mape(confirmed.values[:,1],prophet_pred.Pred)
print(f'Mean Absolute Percentage Error:{MAPE}')


# In[ ]:




