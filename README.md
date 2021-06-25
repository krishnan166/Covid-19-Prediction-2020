# Covid-19 Prediction Using Time Forecasting Models
![Header](https://github.com/krishnan166/Covid-19-Prediction-ARIMA-FB-Prophet/blob/main/covid-19.jpg)

## Inspiration
Given the pandemic scenario, it is important to analyse the trends of cases for better prediction so that disaster management could become a tad bit easy. <br/>

## What it does
This project takes data of three months from Jan-March 2020 worldwide and gives the trend of confirmed, recovered and death cases. It also predicts the upcoming time for the next months as well using two time-forecasting models, ARIMA and FB Prophet models.<br/>

## How I built it
I built the models separately using Jupyter notebook. I took in the same dataset of three CSV files of confirmed, recovered and death cases. Fed the same dataset to both the models to train them and forecast the future. Also, the reliability of the models was checked using performance metrics like root mean squared error, mean absolute error etc. <br/>

## Performance metrics
Root Mean Square Error <br/>
Absolute Mean Error <br/>
Root Relative squared error <br/>
Mean Absolute percentage error <br/>


## What I learned
Both models have their own limitations. However, FB Prophet seems to be a better option than the ARIMA model in this project.<br/>

## What's next for Covid-19 Prediction
Planning to work on including vaccination data for better forecasting.<br/>

