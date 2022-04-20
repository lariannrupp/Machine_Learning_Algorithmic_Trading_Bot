import os
import requests
import pandas as pd

import streamlit as st
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
import hvplot.pandas
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MaxAbsScaler

from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
from finta import TA
import yfinance as yf # place this in your imports section
# something to look into later: https://github.com/kernc/backtesting.py
import time
import itertools
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from functools import reduce
from finta_map import finta_map
import datetime
from collections import Counter

flipped_finta_map = {v: k for k, v in finta_map.items()}

# The wide option will take up the entire screen.
st.set_page_config(page_title="Technical Analysis Machine Learning",layout="wide")
# this is change the page so that it will take a max with of 1200px, instead
# of the whole screen.
st.markdown(
        f"""<style>.main .block-container{{ max-width: 1200px }} </style> """,
        unsafe_allow_html=True,
)
finta_cache = {}


# these are all of the signal indicators provided ba finta. For reference:
# https://github.com/peerchemist/finta
all_ta_functions = [
    'ADL', 'ADX', 'AO', 'APZ', 'ATR', 'BASP', 'BASPN', 'BBANDS', 'BBWIDTH', 
    'BOP', 'CCI', 'CFI', 'CHAIKIN', 'CHANDELIER', 'CMO', 'COPP', 'DEMA', 'DMI', 'DO', 
    'EBBP', 'EFI', 'EMA', 'EMV', 'ER', 'EVWMA', 'EV_MACD', 'FISH', 
    'FRAMA', 'FVE', 'HMA', 'ICHIMOKU', 'IFT_RSI', 'KAMA', 'KC', 'KST', 'MACD', 
    'MFI', 'MI', 'MOBO', 'MOM', 'MSD', 'OBV', 'PERCENT_B', 'PIVOT', 'PIVOT_FIB', 
    'PPO', 'PSAR', 'PZO', 'QSTICK', 'ROC', 'RSI', 'SAR', 'SMA', 'SMM', 'SMMA', 'SQZMI', 
    'SSMA', 'STC', 'STOCH','STOCHRSI', 'TEMA', 'TP', 'TR', 
    'TRIMA', 'TRIX', 'TSI', 'UO', 'VAMA', 'VFI', 'VORTEX', 'VPT', 
    'VWAP', 'VW_MACD', 'VZO', 'WMA', 'WOBV', 'WTO', 'ZLEMA']

# So a bunch of these aren't good. I haven't verified every one of these, but it seems
# like they have very large numbers with large ranges, which means they don't scale well.
# and if they can't scale, they just throw off the data.
bad_funcs = [
    'ADL', 'ADX', 'ATR', 'BBWIDTH', 'BOP', 'CHAIKIN', 'COPP', 'EFI', 'EMV', 'EV_MACD', 
    'IFT_RSI', 'MFI', 'MI', 'MSD', 'OBV', 'PSAR', 'ROC', 'SQZMI', 'STC', 'STOCH', 'ADL', 
    'ADX', 'ATR', 'BBWIDTH', 'BOP', 'CHAIKIN', 'COPP', 'EFI', 'EMV', 'EV_MACD', 'IFT_RSI', 
    'MFI', 'MI', 'MSD', 'OBV', 'PSAR', 'ROC', 'SQZMI', 'STC', 'STOCH', 'UO', 'VORTEX', 'VWAP', 'WTO',
    'WILLIAMS', 'WILLIAMS_FRACTAL', 'ALMA', 'VIDYA','MAMA','LWMA','STOCHD','SWI','EFI']

# Subtracting the known bad functions to make a clearer list
for bad_func in bad_funcs:
    if bad_func in all_ta_functions:
        all_ta_functions.remove(bad_func)
        if bad_func in flipped_finta_map:
            del flipped_finta_map[bad_func]

if 'last_runs_fa_funcs' not in st.session_state:
    # This is initializing the state.
    st.session_state['last_runs_fa_funcs'] = None

def getYahooStockData(ticker, years=10):
    """
    Gets data from yahoo stock api. 
    Args:
        ticker: stock ticker
        years: number of years of data to pull
    Return:
        Dataframe of stock data
    """
    end_date = pd.to_datetime('today').normalize()
    start_date =  end_date - DateOffset(years=years)
    result_df = yf.download(ticker, start=start_date,  end=end_date,  progress=False )

    # renaming cols to be compliant with finta
    result_df = result_df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    
    # dropping un-used col
    result_df = result_df.drop(columns=["Adj Close"])
    return result_df

def prepDf(df):
    """
    Does some basic prep work on the dataframe
    Args:
        df: dataframe to prep
    Return:
        dataframe which has been prepped
    """
    
    df["Actual Returns"] = df["close"].pct_change()
    df = df.dropna()
    # Initialize the new Signal column
    #df['Signal'] = 0.0
    df.loc[:,'Signal'] = 0.0
    # When Actual Returns are greater than or equal to 0, generate signal to buy stock long
    df.loc[(df['Actual Returns'] >= 0), 'Signal'] = 1

    # When Actual Returns are less than 0, generate signal to sell stock short
    df.loc[(df['Actual Returns'] < 0), 'Signal'] = -1
    return df

def makeSignalsDf(ohlcv_df):
    """
    makes the signal df

    Args:
        ohlcv_df: basic ohlcv styled df
    Return:
        dataframe that is date indexed
    """
    signals_df = ohlcv_df.copy()
    signals_df = signals_df.drop(columns=["open", "high", "low", "close", "volume"])
    return signals_df

def executeFintaFunctions(df, ohlcv_df, ta_functions):
    """
    Executes finta functions on a df which is passed in.
    finta reference: https://github.com/peerchemist/finta
    
    Note - so it seems like it's generating these on the fly, which means there's a 
    lot of calculations. Some of these, like DYMI take like 6 seconds to calculate.
    This utilizes a cache variable which is really important in terms of speeding this
    up.

    Args:
        df: a signals df put all of the new cols on
        ohlcv_df: the standard ohlov df
        ta_fuctions: a list of finta functions to call.
    Return:
        dataframe with newly appended finta data.

    """
    
    for ta_function in ta_functions:
        # dynamically calling the TA function.
        try:
            if ta_function in finta_cache:
                # some of these functions are expensive to re-generate. trying to do a
                # cache here to avoid doing the same expensive calculations again and again.
                ta_result = finta_cache[ta_function]
            else:
                # calling the actual finta function:
                # So here, we have the string version of the finta function name that we want
                # to call. This is a way to call a function on an module if just have the string
                # representation of name. the 'getattr' function will return a reference function
                # based on the string name of the function.
                #
                # for example, if we are trying to call the TA.sma() function, at this point we
                # have the 'sma' store as the 'ta_function'. that means that this getattr is
                # returning a reference to TA.sma() without actually calling it, and then storing
                # is as the finta_func variable, which we can then excute on the following line
                # with the ohlcv_df that is necessary to call it.
                finta_func = getattr(TA, ta_function)
                ta_result = finta_func(ohlcv_df)

                finta_cache[ta_function] = ta_result

            # finta functions results vary in terms of data type. Sometimes, it will return
            # a single column of data stored in a panada series. Other times, like with Bollinger
            # bands, it will return three seperate columns of data inside a panda dataframe.
            # this next bit detects what finta returns, and then adds the columns accordingly.
            if isinstance(ta_result, pd.Series):
                df[ta_function] = ta_result
            elif isinstance(ta_result, pd.DataFrame):
                for col in ta_result.columns:
                    df[col] = ta_result[col]
        except Exception as e:
            st.write("Error - failed to execute: ", ta_function)
            st.write("Error - actual error: ", e)
    df.dropna(inplace=True)
    
    indicators=list(df.columns)
    indicators.remove("Actual Returns")
    indicators.remove("Signal")

    return (df, indicators)

def createScaledTestTrainData(df, indicators, scaler_name):
    """
    created scaled training and test data.

    Args:
        df: data frame
        indicators: all of the indicator data to scale
    Return:
        tuple(X_train_scaled, X_test_scaled, y_train, y_test)
    """
    X = df[indicators].shift().dropna()
    y = df['Signal']
    y=y.loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, shuffle=False)

    # creating the actual scaler based on the scaler_name that is passed in.
    scaler = None
    if scaler_name == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_name == 'MinMaxScaler':
        scaler = MinMaxScaler(feature_range=(-1,1))
    elif scaler_name == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    elif scaler_name == 'PowerTransformer':
        scaler = PowerTransformer()
    elif scaler_name == 'QuantileTransformer':
        scaler = QuantileTransformer(output_distribution="normal")
    elif scaler_name == 'RobustScaler':
        scaler = RobustScaler()

    # Apply the scaler model to fit the X-train data
    #X_scaler = scaler.fit(X_train)
    # Transform the X_train and X_test DataFrames using the X_scaler
    #X_train_scaled = X_scaler.transform(X_train)
    #X_test_scaled = X_scaler.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def executeSVMModel(X_train_scaled, X_test_scaled, y_train, y_test, signals_df ):
    """
    executs the svm model on the data provided

    Args:
        X_train_scaled: scaled training dataset
        X_test_scaled: scaled test dataset
        y_train: scaned training dataset
        y_test: scaled test dataset
        signals_df: signals df for the 'actual returns' col and index
    Return:
        tuple(predictions_df, testing_report)
    """
    
    model = svm.SVC()
    model = model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    testing_report = classification_report(y_test, pred, output_dict=True)

    #st.write(X_test_scaled)
    #if len(np.unique(pred)) == 1:
    #st.write(">>> hit pred count:", pred)
    #st.write(Counter(pred).keys())
    #st.write(Counter(pred).values())    
    
    
    predictions_df = pd.DataFrame(index=y_test.index)
    # Add the SVM model predictions to the DataFrame
    predictions_df['Predicted'] = pred

    # Add the actual returns to the DataFrame
    predictions_df['Actual Returns'] = signals_df['Actual Returns']

    # Add the strategy returns to the DataFrame
    predictions_df['Strategy Returns'] = (predictions_df['Actual Returns'] * predictions_df['Predicted'])
    return predictions_df, testing_report

def executeRandomForest(X_train_scaled, X_test_scaled, y_train, y_test, signals_df):
    """
    executs the random forest on the data provided

    Args:
        X_train_scaled: scaled training dataset
        X_test_scaled: scaled test dataset
        y_train: scaned training dataset
        y_test: scaled test dataset
        signals_df: signals df for the 'actual returns' col and index
    Return:
        tuple(predictions_df, testing_report)
    """
    #rf_model = RandomForestClassifier(n_estimators=100)
    rf_model = RandomForestClassifier()
    rf_model = rf_model.fit(X_train_scaled, y_train)
    pred = rf_model.predict(X_test_scaled)
    testing_report = classification_report(y_test, pred, output_dict=True)

    predictions_df = pd.DataFrame(index=y_test.index)
    # Add the SVM model predictions to the DataFrame
    predictions_df['Predicted'] = pred

    # Add the actual returns to the DataFrame
    predictions_df['Actual Returns'] = signals_df['Actual Returns']

    # Add the strategy returns to the DataFrame
    predictions_df['Strategy Returns'] = (predictions_df['Actual Returns'] * predictions_df['Predicted'])
    return predictions_df, testing_report

def executeNaiveBayes(X_train_scaled, X_test_scaled, y_train, y_test, signals_df):
    """
    executs the naive bayes on the data provided

    Args:
        X_train_scaled: scaled training dataset
        X_test_scaled: scaled test dataset
        y_train: scaned training dataset
        y_test: scaled test dataset
        signals_df: signals df for the 'actual returns' col and index
    Return:
        tuple(predictions_df, report)
    """
    
    model = GaussianNB()
    model = model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    report = classification_report(y_test, pred, output_dict=True)

    predictions_df = pd.DataFrame(index=y_test.index)
    # Add the SVM model predictions to the DataFrame
    predictions_df['Predicted'] = pred

    # Add the actual returns to the DataFrame
    predictions_df['Actual Returns'] = signals_df['Actual Returns']

    # Add the strategy returns to the DataFrame
    predictions_df['Strategy Returns'] = (predictions_df['Actual Returns'] * predictions_df['Predicted'])
    return predictions_df, report


def execute(ticker, scaler, indicators_to_use=[], years=10, rerun=False):
    """
    This is the main data gathering for this app. It will call other functions
    to assemble a main dataframe which can be used in different ways.
    Args:
        ticker: ticker to use
        indicators_to_use: indicators to use
        years: # of years of data to base all of this on.
    Return:
        None
  
    """

    # Getting the stock data
    ohlcv_df = getYahooStockData(ticker.upper(), years)

    #prepping the stock data
    ohlcv_df = prepDf(ohlcv_df)

    ta_functions = random.choices(all_ta_functions, k=5)
    if indicators_to_use:
        ta_functions = indicators_to_use
    elif rerun == True:
        ta_functions = st.session_state['last_runs_fa_funcs']
        
    st.session_state['last_runs_fa_funcs'] = ta_functions

    names = [flipped_finta_map[n] for n in ta_functions]

    #this is generating all of the combinations of the ta_functions. 
    ta_func_combinations = []
    for k in range(len(ta_functions)):
        ta_func_combinations.extend(itertools.combinations(ta_functions, k+1))

    st.write(f"Testing {len(ta_func_combinations)} different combinations of these indicators: ", ", ".join(names))
    
    # this is prepping the final results df    
    top_ten_results_df = pd.DataFrame(columns=["Variation", "SVM Returns", "Random Forest Returns", "Naive Bayes Returns"])

    # all of the results dfs should be stored in this map for future reference
    all_combinations_result_map = {}

    # this is really important. some of the finta functions 
    # take a long time. having a cache really speeds it up
    finta_cache = {}

    actual_returns_for_period = None
    
    for ta_func_permutation in ta_func_combinations:

        perm_key = ",".join(ta_func_permutation)

        # it's lame to do this every time, but I've experienced so many reference errors with not
        # trying to re-instiate this every loop. Yes, it's lame, but this is a bit more garenteed
        # to work.
        signals_df = makeSignalsDf(ohlcv_df)
        finta_signals_df, indicators = executeFintaFunctions(signals_df, ohlcv_df, ta_func_permutation)

        X_train_scaled, X_test_scaled, y_train, y_test = createScaledTestTrainData(finta_signals_df, indicators, scaler)

        svm_predictions_df, svm_testing_report = executeSVMModel(X_train_scaled, X_test_scaled, y_train, y_test, signals_df)
        rf_predictions_df, rf_testing_report = executeRandomForest(X_train_scaled, X_test_scaled, y_train, y_test, signals_df)
        nb_predictions_df, nb_testing_report = executeNaiveBayes(X_train_scaled, X_test_scaled, y_train, y_test,signals_df)

        svm_final_df = (1 + svm_predictions_df[['Actual Returns', 'Strategy Returns']]).cumprod()
        rf_final_df = (1 + rf_predictions_df[['Actual Returns', 'Strategy Returns']]).cumprod()    
        nb_final_df = (1 + nb_predictions_df[['Actual Returns', 'Strategy Returns']]).cumprod()
        
        # at this point we have all of our results. This next bit is a way to rename the different cols
        # and then merge them into a single dataframe which we can use to chart later in the results.
        rf_final_df.drop(columns=['Actual Returns'], inplace=True)
        nb_final_df.drop(columns=['Actual Returns'], inplace=True)

        svm_final_return = svm_final_df.iloc[-1]["Strategy Returns"]
        rf_final_return = rf_final_df.iloc[-1]["Strategy Returns"]
        nb_final_return = nb_final_df.iloc[-1]["Strategy Returns"]

        svm_final_df.rename(columns={'Strategy Returns': 'SVM Returns'}, inplace=True)
        rf_final_df.rename(columns={'Strategy Returns': 'Random Forest Returns'}, inplace=True)
        nb_final_df.rename(columns={'Strategy Returns': 'Naive Bayes Returns'}, inplace=True)        

        dfs_to_merge = [svm_final_df, rf_final_df, nb_final_df]
        merged_df = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True), dfs_to_merge)
       
        _key = ",".join([flipped_finta_map[n] for n in ta_func_permutation])

        # now that all of the results are in a single dataframe, we're storing the merged_df in a map so that
        # it could possibly be referenced later to display the chart later.
        all_combinations_result_map[_key] = merged_df        

        # the next 3 lines is a way to manually add a row to a dataframe
        top_ten_results_df.loc[-1] = [_key, svm_final_return, rf_final_return, nb_final_return]
        top_ten_results_df.index = top_ten_results_df.index + 1
        top_ten_results_df = top_ten_results_df.sort_index()

        # fixme - it's lame to have to reset this every iteration
        actual_returns_for_period = svm_final_df.iloc[-1]["Actual Returns"]

    top_ten_results_df = top_ten_results_df.sort_values(by=["SVM Returns", "Random Forest Returns", "Naive Bayes Returns"], ascending=False)

    st.write(f"Return for {ticker} over the testing period is {round(actual_returns_for_period,4)}")
    st.write("Top 10 Models:")

    hide_table_row_index = """<style>tbody th {display:none} .blank {display:none} </style> """

    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)    

    st.table(top_ten_results_df.head(10))
    
    for perm_key in top_ten_results_df["Variation"][:10]:
        st.write(f"Results for: {perm_key}")
        st.line_chart(all_combinations_result_map[perm_key])
   
   # Display classification report
    with st.expander("Classification Report Comparison"):
        st.write('SVM Report')
        st.table(pd.DataFrame(svm_testing_report))
        st.write('RandomForest Report')
        st.table(pd.DataFrame(rf_testing_report))
        st.write('Naive Bayes')
        st.table(pd.DataFrame(nb_testing_report))

def main():
    """
    Main function of this app. Sets up the side bar and then exectues the rest of the code.

    Returns:
        None
    """
   
    st.title("Technical Indicator Analysis with ML")

    st.sidebar.info( "Select the criteria to run:")

    # reversing this again
    valid_indicators = {v: k for k, v in flipped_finta_map.items()}

    valid_indicator_names = valid_indicators.keys()

    all_scalers = ['StandardScaler','MinMaxScaler','MaxAbsScaler',
                   'QuantileTransformer','PowerTransformer', 'RobustScaler']
    
    selected_stock = st.sidebar.text_input("Choose a stock:", value="SPY")
    selected_scaler = st.sidebar.selectbox("Choose a Scaler:", all_scalers)
    
    st.sidebar.markdown("---")    
    named_selected_indicators = st.sidebar.multiselect("TA Indicators to use:", valid_indicator_names)

    selected_indicators = []

    # It would be interesting to play with date ranges. 
    # selected_years = st.sidebar.slider("Number of years of data", min_value=1, max_value=10, value=10)
    selected_years = 10
    
    for named_indicator in named_selected_indicators:
        selected_indicators.append(valid_indicators[named_indicator])

    if st.sidebar.button("Run"):
        with st.spinner('Executing...'):        
            execute(selected_stock, selected_scaler, selected_indicators, selected_years)
    st.sidebar.markdown("---")
    st.sidebar.write("This will randomly choose 5 indicators")
    if st.sidebar.button("I'm feeling lucky"):
        with st.spinner('Executing...'):
            execute(selected_stock, selected_scaler)
    if st.sidebar.button("Re-run last"):
        with st.spinner('Executing...'):
            execute(selected_stock, selected_scaler, [],10, True)
        
main()
