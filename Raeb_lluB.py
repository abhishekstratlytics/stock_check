import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt 
#import plotly  
#from plotly import graph_objs as go 
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from statsmodels.tsa.arima.model import ARIMA
import datetime
# import pmdarima as pm
# from sklearn.metrics import mean_absolute_error as mae
# import statsmodels.api as sm
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.stattools import acf,pacf
import itertools 


### Naming the app with an icon
img = Image.open('Raeb_lluB.jpg')
st.set_page_config(page_title = "Raeb_lluB",
    page_icon=img)

## Functions for creating the app

### Checks for stationarity and returns the order of differencing required


## The main function

def main():
    st.title("Raeb_lluB")
    st.image("Raeb_lluB.jpg",width=300)
    st.subheader("A Bull Bear Affair")
    nav= st.sidebar.radio("Navigation",["HOME","PREDICTION","PORTFOLIO"])

    ## Loading the data
    uploaded_file = st.file_uploader("File_Upload")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("prices.csv")

    ### Pre-processing
    #rp = pd.read_csv('prices.csv') #raw prices
    data.rename(columns={ data.columns[1]: "Date" }, inplace = True)
    rp = data.copy()
    rp['Date']=pd.to_datetime(rp['Date'])
    rp = rp[['Date','SYMBOL',' CLOSE_PRICE']]
    processed_prices = rp.pivot_table(index='Date',columns='SYMBOL',values=' CLOSE_PRICE')
    processed_prices = processed_prices.dropna(axis=1)
    stock_info = pd.read_csv('ind_nifty500list.csv')
    Selected_stocks = stock_info['Symbol'].to_list()
    orignalstocks = processed_prices.columns
    def common_member(a, b):
        result = [i for i in a if i in b]
        return result
    common_stocks = common_member(orignalstocks,Selected_stocks)
    #exclusion = ['360ONE', 'ACI', 'ADANIPOWER', 'AETHER', 'AWL', 'BCG', 'BEML', 'BIKAJI', 'CAMPUS', 'DELHIVERY', 'FIVESTAR', 'FLUOROCHEM', 'JINDWORLD', 'KFINTECH', 'LICI', 'LTIM', 'MEDANTA', 'MOTHERSON', 'NSLNISP', 'PPLPHARMA', 'RAINBOW', 'RUSTOMJEE', 'SHRIRAMFIN', 'TMB', 'TTML', 'UNOMINDA']
    res = [i for i in common_stocks]
    processed_prices = processed_prices[res]
    processed_prices.to_csv('processed_prices.csv',index = True)
    #st.write(processed_prices.head())
    stock_info = stock_info[stock_info['Symbol'].isin(res)] #Filtering stocks which are not present in orignal list
    #stock_info = stock_info.reset_index(drop=True)
    #stock_info.to_csv('Stock_info.csv')
    
    # Moving average prediction Model
    a = st.text_input("Enter for how many days do you want to calculate the Moving Average",value = 10)
    Rolling_Width = int(a)
    # Read processed Stock prices
    stocks_prices = pd.read_csv('processed_prices.csv')
    #stocks_prices = processed_prices.copy()
    stocks_prices_len = stocks_prices.shape[0] -1
    stocks_prices = stocks_prices[stocks_prices['Date'] == stocks_prices['Date'][stocks_prices_len]]
    output_df = stocks_prices.transpose()
    date = output_df[stocks_prices_len][0]
    output_df = output_df.iloc[1:]
    output_df.reset_index(inplace = True)
    output_df['Date'] = date
    output_df.columns = ["Symbol", "Price", "Date"]
    #stock_info = pd.read_csv('Stock_info.csv')
    stock_info = stock_info.copy()
    stock_info = stock_info[['Symbol', 'Industry']]
    output_df = output_df.merge(stock_info, on = "Symbol", how = "left")
    output_df = output_df[['Date', 'Symbol', 'Industry', 'Price']]
    ## Adding Moving averages for n days(user input)

    stocks_prices1 = pd.read_csv('processed_prices.csv')

    # Cretaing an empty dataframe to store the Moving average values
    stocks_prices2 = pd.DataFrame()
    stocks_prices2['Date'] = stocks_prices1['Date']
    # Setting date as index
    stocks_prices1 = stocks_prices1.set_index('Date')
    stocks_prices2 = stocks_prices2.set_index('Date')
    cols = stocks_prices1.columns

    for i in range(stocks_prices1.shape[1]):
        stocks_prices2[cols[i]] = stocks_prices1[cols[i]].rolling(Rolling_Width).mean()

    stocks_prices2.reset_index(inplace = True)
    stocks_prices2 = stocks_prices2.tail(1)
    stocks_prices2 = stocks_prices2.transpose()

    stocks_prices2 = stocks_prices2.iloc[1:]
    stocks_prices2.reset_index(inplace = True)
    stocks_prices2.reset_index(inplace = True,drop = True)
    stocks_prices2.columns = ['Symbol', (str(Rolling_Width)+ '_MA')]

    # # output_df = pd.read_csv('Predicted_Price.csv').drop({'Unnamed: 0'},axis = 1)
    output_df = output_df.merge(stocks_prices2, on = 'Symbol', how = "left")
    output_df.drop(columns = ["Price"],inplace = True)
    output_df.rename(columns = {stocks_prices2.columns[1]:'Price'},inplace =True)
    output_df.to_csv("Predicted_Price.csv", index = False)
    

    ## Output 
    # read the predicted price output submited by Padmaja
    df=pd.read_csv('Predicted_Price.csv')
    # read the preprocessed data submited by Raj
    df2=pd.read_csv('processed_prices.csv')
    df2['date']=df2['Date']
    df2.drop('Date',axis=1,inplace=True)
    df2.sort_values(by='date', ascending=True)
    numeric_cols = df2.select_dtypes(exclude=['object']).columns
    percentage_change = ((df2[numeric_cols].iloc[-1] - df2[numeric_cols].iloc[0]) / df2[numeric_cols].iloc[-1]) * 100
    #percentage_change
    df3=percentage_change.to_frame()
    columns=['Symbol','return']
    df3 = df3.reset_index(drop=False)
    df3.columns=['Symbol','return']
    merged_df = pd.merge(df, df3, on='Symbol', how='left')
    df1=merged_df.copy()
    b = st.text_input("Enter for how many days do you want to calculate the Moving Average",value = 1000000)
    total_investment= float(b)

    #df1['Price Criteria'] = df1['Price'] < (0.05 * total_investment)
    #df1['Return Criteria'] = df1['return'] > 0

    df1=df1[df1['Price']<=0.05*total_investment]
    df1=df1[df1['return']>0]
    industry_groups = df1.groupby('Industry')['Symbol'].count()
    df = df1.sort_values('return', ascending=False)
    df=df[['Symbol', 'Industry', 'Price', 'return']]
    sectors={key:0 for key in set(df['Industry'])}
    selected_symbols = []
    sectors={key:0 for key in set(df['Industry'])}
    total_price = 0
    for symbol, row in df.iterrows():
        #print(row)
        #break
        if row['Price']+total_price <=total_investment:
            
            if sectors[row['Industry']]+row['Price']<=0.20*total_investment:
                
                sectors[row['Industry']]=row['Price']
                selected_symbols.append((row['Symbol'],int(0.5*total_investment//row['Price']),row['Price']))
                total_price+=row['Price']
    def optimalSolution(df:pd.DataFrame)->list:
        selected_symbols = []
        sectors={key:0 for key in set(df['Industry'])}
        total_price = 0
        for symbol, row in df.iterrows():
        #print(row)
        #break
            if row['Price']+total_price <=total_investment:
                
                if sectors[row['Industry']]+row['Price']<=0.20*total_investment:
                
                    sectors[row['Industry']]=row['Price']
                    selected_symbols.append((row['Symbol'],int(0.5*total_investment//row['Price']),row['Price'],row['Industry']))
                    total_price+=row['Price']
        return selected_symbols,sectors
    sol,sectors=optimalSolution(df)

    sol=pd.DataFrame(sol, columns=['Symbol','No_of_sysmbols','Amount_invested_in_Symbol','Industry']).sort_values('Amount_invested_in_Symbol',ascending=False)
    sol = sol.sort_values("Amount_invested_in_Symbol",ascending = False)
    sector={'Industry':sectors.keys(),
        'TotalAmount':sectors.values()}
    sector= pd.DataFrame(sector)
    sector= sector.sort_values("TotalAmount",ascending = False)
    sector=sector[sector['TotalAmount']>0]




    if nav == "HOME":
        st.subheader("HOME PAGE")
        st.subheader("Imported Data")
        st.write(rp.head())
        st.subheader("Processed Data")
        st.write(processed_prices.tail(5))


    elif nav == "PREDICTION"  :

        st.header("Your Product Prediction")
        st.write(output_df.head(5))
        
        #col1,col2 = st.columns(2)
        #with col1:
            #st.write("Data used for building the model")
            #st.write(a)
        #with col2:
            #st.write("Data used for testings the model")
            #st.write(b)
        
        st.download_button("Download Output",output_df.to_csv(),file_name = "All_stock_prediction.csv",mime='text/csv')


    else:
        st.subheader("Portfolio Prediction")
        sol = sol.round(decimals=0)
        st.write(sol)
        sector = sector.round(decimals=0)
        st.write(sector)
        
        st.download_button("Portfolio",sol.to_csv(),file_name = "Portfolio.csv",mime='text/csv')
        st.download_button("Portfolio_Sector",sector.to_csv(),file_name = "Sector.csv",mime='text/csv')



if __name__ == '__main__':
	main()
