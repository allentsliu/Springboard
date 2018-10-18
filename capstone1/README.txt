1. What is the problem you want to solve?
Because of the volatility of the stock market, it is very difficult to predict the value of a stock market without spending a considerable about of time researching the stock.

2. Who is your client and why do they care about this problem? In other words, what will your client DO or DECIDE based on your analysis that they wouldn’t have otherwise?
Everyone who is interested in investing stock can use this stock predictor program. 

3. What data are you going to use for this? How will you acquire this data?
Data set from quandl. Calling quandl API

4. In brief, outline your approach to solving this problem (knowing that this might change later).
To develop a stock predictor, I will use the data set from quandl for training different kind of ML models. 

5.What are your deliverables? Typically, this would include code, along with a paper and/or a slide deck.
Code and README file

Data wrangling steps to clean the data set:
Once the data is loaded into the data frame, df, df.head() is used to get a general understand of the data.
Then, df.describe() is used to make sure that there is no outliner in the data.
To verify whether there is any missing data, df.info() is used.
As shown below, there is no missing data
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 4804 entries, 1999-01-22 to 2018-02-26
Data columns (total 5 columns):
adj_open      4804 non-null float64
adj_high      4804 non-null float64
adj_low       4804 non-null float64
adj_close     4804 non-null float64
adj_volume    4804 non-null float64
dtypes: float64(5)
memory usage: 225.2 KB

To create the target value, a new column, y_30, is created by shifting the adj_close column 30 cells up. df['y_30'] = df['adj_close'].shift(periods=30)
The purpose of the shift is to simulate future stock price. 

After the shift, the rows with empty values are removed using df.dropna(). 
After splitting the data to training set and test set, StandardScaler() is used to scale the data before the training.
