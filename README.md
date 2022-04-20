![Boom-Bull-Stock-Exchange-Bear-World-Economy-913982](https://user-images.githubusercontent.com/95719899/164332595-9cf0e024-aab1-4d3b-a11c-8c47ee122b9e.jpg)

---
# Machine_Learning_Algorithmic_Trading_Bot

**LAUNCH APP**: https://share.streamlit.io/lariannrupp/machine_learning_algorithmic_trading_bot/main/streamlit.py

---
## Purpose: 

Before starting their algorithmic trading project or purchasing a trading bot from an online marketplace, a user can launch this app to select a stock and compare machine learning models and technical indicators in a generic testing environment. 



---

## Installation Guide

The app is deployed to a Streamlit interface, so to use the app, no installation is required. Simply launch the link at the top of the README. 

---

## Data

10 years of OHLVC (open, high, low, volume, close) stock data is pulled from the Yahoo Finance API. 

The data undergoes a standard train, fit, predit process with the training window being 6 years, and the testing window being 4 years. 

By default, the app scales X feature with StandardScaler(), but within the app, the user can test out different data scaling methods. 

---

## Selected Machine Learning Models

The following machine learning models were selected because they are common, supervised models with binary classification systems:

- SVM (Support Vector Machine)

- Random Forest

- Naive Bayes

- AdaBoost 

- Decision Tree

---

## Approach

The user can test combinations of up to 5 technical indicators. Of the 31 possible combinations, the top 10 are displayed. 

For users who would like to explore random combinations of 5 indicators, they can use the **I'm feeling lucky** button. 

The **Re-run last** button allows the user the rerun the app observe how the random forest model performs with each iteration.


---

## Performance Evaluation and Backtesting

Cumulative return plots compare model performance to market performance. Additionally, the table for "Top 10 Models" compares cumulative returns as both a ratio and a percentage. 


![Screenshot 2022-04-20 171355](https://user-images.githubusercontent.com/95719899/164341560-ee00d663-34b1-4df4-81a2-f6c466ac306f.jpg)



To backtest the models and compare trade accuracies, the user can drop down the **Classification Report Comparison** button. Please note that trade accuracies can oftentimes be a better metric of model performance than cumulative returns. 

![Screenshot 2022-04-20 171609](https://user-images.githubusercontent.com/95719899/164341569-c4fedbb2-2749-48a9-b6b7-c4e8433a484f.jpg)

---

## Contributors:

### Development Team
Leigh Badua

John Batarse

Catherine Croft

Jing Pu

Jason Rossi

Lari Rupp


### Collaborators
University of California Berkelely Fintech Bootcamp (Project 2, 2022),

Kevin Lee (Instructor)

Vincent Lin (TA)

---


## Technologies

A Python 3.9.7 (ipykernal) was used to write this app.

![python-logo-master-v3-TM-flattened](https://user-images.githubusercontent.com/95719899/164334658-d32c6762-b35d-4ae3-8d87-f054388941e7.png)
![Pandas_logo svg](https://user-images.githubusercontent.com/95719899/164334292-8243632d-1274-4c4f-ba36-cbf71dc14309.png)
![YahooFinance](https://user-images.githubusercontent.com/95719899/164334383-5f613f77-fb14-4b8c-80a7-882241baf76a.png)
![1200px-Finta_Logo](https://user-images.githubusercontent.com/95719899/164334464-705a5167-9385-4f93-91b4-5afc74a0ea24.png)
![1200px-Scikit_learn_logo_small svg](https://user-images.githubusercontent.com/95719899/164334470-dac38a18-1d42-4bfe-abfe-7f681677a8ff.png)
![streamlit_logo](https://user-images.githubusercontent.com/95719899/164334479-b14755bc-7525-4f9b-aeaf-6e56df94f49d.png)


---

## License

Creative Commons Zero

This is a truly open-source project under the Creative Commons Zero license which is free for use for everyone.

We ask that you please credit the team with the following IEEE citation:

> L. Badua, J. Batarse, C. Croft, J. Pu, J. Rossi, L. Rupp, “Machine_Learning_Algorithmic_Trading_Bot,” University of California Berkeley Extension Fintech Bootcamp, Berkeley, California, USA, Fintech Bootcamp Project 2, 2022. https://github.com/lariannrupp/Machine_Learning_Algorithmic_Trading_Bot (accessed month day, year).
