##
## pipreqs (to create requirement file needed for streamshare)
## cd to current folder 
## streamlit run GBM_simulation_Streamlit.py

########################## Initialization - Lib and Settings #####################

import streamlit as st
from streamlit_autorefresh import st_autorefresh

import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import time
from os import path
 

import yfinance as yf

import matplotlib.pyplot as plt
import plotly.express as px 




########################## Streamlit Settings #####################
st.set_page_config(layout="wide")


# Graph properties
background = {
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
}
grid_thickness = 1;
grid_colour = "grey"
background_colour = "black"
blue_color = '#0000ff'
red_color = "#ff0000"
width_line= 3
shape_height = 300
shape_width = 900



def line_chart(df,Title,x_title,y_title,height):

	line_chart = px.line(df,title=Title,height = height)

	line_chart.layout.plot_bgcolor= background_colour
	line_chart.update_traces( line_width=width_line)
	line_chart.update_xaxes( gridwidth=grid_thickness,gridcolor=grid_colour)
	line_chart.update_yaxes( gridwidth=grid_thickness,gridcolor=grid_colour)
	line_chart.layout.yaxis.title=y_title
	line_chart.layout.xaxis.title=x_title

	return line_chart

# Create a function for GBM simulation 
import numpy as np 

@st.cache_data
def simulate_gbm(S0, mu,sigma, days, num_simulations):
    
    # Time increment (daily)
    ## As we have daily returns - instead of scaling time,mu,sigma statistically, 
    ## Lets just make this simple by always having increament of one day.             
    dt = 1 
    
    ## Creating simulation array of number of days * num of simulations 
    simulations = np.zeros((days+1, num_simulations))
    
    ## # Calculating Random component assuming normal distribution
    wt_array = np.random.normal(size = (days,num_simulations))
    
    for i in range(num_simulations):  # for every simulation
        
        price_path = [S0] # initial price
                
        for j in range(days): # for each day 
            
                
            Wt = wt_array[j,i]  
            price = price_path[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * Wt)
            price_path.append(price)
            
            
        simulations[:, i] = price_path[:]
        
    return simulations


@st.cache_data
def data_download(YF_ticker):
	# Get data
	start_data = datetime.datetime.now().date() - timedelta(365*10)#w'2001-01-01'
	end_date = datetime.datetime.now().date() #'2021-12-31'
	Data_Prices=yf.download(YF_ticker,start=start_data, end=end_date)
	# Data_Prices.reset_index(inplace = True)

	return Data_Prices

#########################################################


st.title("Geometric Brownian Motion Simulations")

## Input parameters


YF_ticker_list = ('^GSPC', 'AAPL', 'GOOG','MSFT','TSLA') #"^GSPC" 


st.divider()


col_1, col_2 = st.columns([1,2])
with col_1:
	st.write("")
	st.subheader("Historical Performance")

	st.write("")
	YF_ticker = st.selectbox('Select Instrument for analysis',YF_ticker_list)
	st.write("")
	st.write('You selected:', YF_ticker)

	Data_Prices = data_download(YF_ticker)

	# Calculate daily returns percentage
	Data_Prices['Daily Return'] = Data_Prices['Adj Close'].pct_change().dropna()

	# Estimate drift (mean of daily returns)
	estimated_drift = (Data_Prices['Daily Return'].mean() )

	# Estimate volatility (standard deviation of daily returns)
	estimated_volatility = Data_Prices['Daily Return'].std() 

	
	st.divider()
	st.write("##### Analysis")
	st.write("Mean of daily % returns mean i.e. (drift) =", round(estimated_drift* 100,4) , " %")
	st.write("Std dev of daily % returns i.e. (volatility) = ", round(estimated_volatility* 100,4) , " %")


	

with col_2:
	close_chart  = line_chart(Data_Prices.Close,"10 Year Price Movement","Date","Close_Price",500)# line_chart("Close Price",Data_Prices,"Date","Close")

	st.plotly_chart(close_chart, use_container_width=True)


st.divider()

col_1, col_2 = st.columns([1,2])



with col_1:

	col_11, col_21 = st.columns([1,2])
	with col_11:
		st.write("")
		st.subheader("Simulations")

	with col_21:
		st.write("")
		state = st.button("Simulate Prices", type="primary")
		st.write(state)


with col_1:		
	st.write("")
	num_sim = st.slider('Number of Simulations?', 1, 1000, 100 , 1)
	# st.write("You selected: ", num_sim)

	st.write("")
	num_days = st.slider('Number of Days for simulation?', 1, 504, 252 , 1)
	# st.write("You selected: ", num_days)

	st.write("")
	S0 = Data_Prices.Close.iloc[-1]
	mu =  estimated_drift *100
	sigma = estimated_volatility * 100

	mu_selected = st.slider(' Simulation Drift [mu-sigma to mu+sigma]', mu - sigma , mu + sigma, mu , 0.001)
	mu_selected = round(mu_selected,4)
	# st.write("You selected: ", mu_selected)
	mu_simulation = mu_selected/100

	st.write("")
	sigma_selected = st.slider(' Simulation Volatility [0 to 2*sigma]', 0.00000, 2 * sigma , sigma , sigma/10) 
	sigma_selected = round(sigma_selected,4)
	# st.write("You selected: ", sigma_selected, "% of historical volatility")
	sigma_simulation = sigma_selected * sigma/100

with col_2:

	if state:

		GBM_Simulation = simulate_gbm(S0, mu_simulation,sigma_simulation, num_days, num_sim)
		
		st.write("Simulated Graph")
		simulated_chart  = line_chart(GBM_Simulation,"Simulated Price Movement","Date","Close_Price",600)
		st.plotly_chart(simulated_chart, use_container_width=True)

		st.session_state.simulation = GBM_Simulation


	elif 'simulation' not in st.session_state :

		st.write ("Waiting for user to click simulate")
		

	if (state == False) and ('simulation' in st.session_state):

		st.write("Waiting for user to click Simulate price and update the graph")
		simulated_chart  = line_chart(st.session_state.simulation,"Simulated Price Movement","Date","Close_Price",600)
		st.plotly_chart(simulated_chart, use_container_width=True)


st.divider()

"## Code"

"For detail code refer my github repo :- "
" 1. [[webApp]](https://quantproject1-csovwwndasw9kuk2vpygjp.streamlit.app/) [Predictive Analysis of Stock Trajectories using Geometric Brownian Motion](https://github.com/Kapil3003/Quant_Project_1/blob/main/Project_1_GBM.ipynb)"

" 2. [Comprehensive VaR Analysis: Methods Comparison, Backtesting, and Stress Testing](https://github.com/Kapil3003/Quant_Project_2/blob/main/Project_2_VaR_Analysis.ipynb)"

" 3. [Robust Trading Strategy Development using Walk Forward Optimization](https://github.com/Kapil3003/Quant_Project_3/blob/main/Project_3_StrategyDevelopment.ipynb)"

" 4. [Market Volatility Forecasting: An ExtensiveComparative Study](https://github.com/Kapil3003/Quant_Project_4/blob/main/Project_4_Volatility%20Forecasting.ipynb)"

" 5. [[webApp]](https://quantproject5-gcs2rtyqub8wj8osxwegu2.streamlit.app/) [Real-Time Options Chain Data Analysis Dashboard](https://github.com/Kapil3003/Quant_Project_5)"



