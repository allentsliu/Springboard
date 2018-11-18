import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import model_from_yaml
from sklearn.externals import joblib
import os
import quandl
import collections
import pickle
import datetime



def download_model(ticker, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(ticker+'.yaml'):
    if not os.path.exists('data/'+ticker+'.csv'):
        download_data(ticker)
    df = pd.read_csv('data/'+ticker+'.csv', usecols=['date', 'adj_close', 'adj_volume'], delimiter='\t', header=0,
                     index_col='date',
                     parse_dates=True)
    data = clean_data(df)
    model = train(data.x_train, data.y_train, data.x_test, data.y_test, ticker)
  else:
      # load json and create model
      json_file = open(ticker+'.yaml', 'r')
      loaded_model_yaml = json_file.read()
      json_file.close()
      model = model_from_yaml(loaded_model_yaml)
      # load weights into new model
      model.load_weights(ticker+".h5")
      print("Loaded model from disk")
  return model


def download_data(ticker):
    data = quandl.get_table("WIKI/PRICES", qopts={
        'columns': ['ticker', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']}, ticker=ticker,
                            paginate=True)
    if data.shape[0] > 1:
        data.to_csv('data/' + ticker + '.csv', '\t')

def clean_data(df):
    df['y_30'] = df['adj_close'].shift(periods=30)
    df = df.dropna()
    x_train = df.iloc[:3700, 0:-1].values
    # print(x_train)
    y_train = df.iloc[:3700, -1].values
    # print(y_train)
    x_test = df.iloc[3700:, 0:-1].values
    y_test = df.iloc[3700:, -1].values
    # print(y_test)

    scale = StandardScaler()
    x_train = scale.fit_transform(x_train)
    x_test = scale.transform(x_test)
    joblib.dump(scale, scaler_filename)
    data = collections.namedtuple('data', ['x_train', 'y_train', 'x_test', 'y_test'])
    return data(x_train, y_train, x_test, y_test)

def train(x_train, y_train, x_test, y_test, ticker):
    model = Sequential()
    model.add(Dense(32, input_shape=(2,)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, batch_size=10)
    score, acc  = model.evaluate(x_train, y_train, batch_size=10, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], acc * 100))
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(ticker+".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(ticker+".h5")
    print("Saved model to disk")
    return model


quandl.ApiConfig.api_key = "nVD4QZoCjEQijoM1Pvzz"
scaler_filename = "scaler.save"
model = download_model('AAPL')

end_date = datetime.date.today()
thirty_day = datetime.timedelta(days=30)
start_date = end_date - thirty_day

dataset_data = quandl.Dataset('EOD/AAPL').data(params={ 'start_date':start_date, 'end_date':end_date })

# dataset_data = quandl.Dataset('WIKI/AAPL').data(params={ 'start_date':'2001-01-01', 'end_date':'2010-01-01', 'collapse':'annual', 'transformation':'rdiff', 'rows':4 })

input_data = dataset_data.to_pandas()
input_data1 = input_data[['Adj_Close','Adj_Volume']].values
scale = joblib.load(scaler_filename)
input_data1 = scale.transform(input_data1)
predicted_data = model.predict(input_data1)
plt.plot()
plt.plot(predicted_data)
plt.show()







# print(df.head())






# Cs = [0.001, 0.01, 0.1, 1, 10]
# gammas = [0.001, 0.01, 0.1, 1]
# param_grid = {'C': Cs, 'gamma': gammas}
# grid_search = GridSearchCV(SVR(), param_grid)
# grid_search.fit(x_train, y_train)
# print(grid_search.best_params_)

# clf = SVR(C=10, gamma=0.001).fit(x_train, y_train)
# train_score = clf.score(x_train, y_train)
# test_score = clf.score(x_test, y_test)
# print("train score {}".format(train_score))
# print("test score {}".format(test_score))



# y_predicted = model.predict(x_test)
#
# print(len(y_predicted))
# print(len(y_test))
#
# plt.plot(y_test)
# plt.plot(y_predicted)
# plt.show()