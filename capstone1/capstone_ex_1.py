import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


df = pd.read_csv('data/NVDA.csv', usecols=['date', 'adj_close', 'adj_volume'], delimiter='\t', header=0, index_col='date',
                 parse_dates=True)

df['y_30'] = df['adj_close'].shift(periods=10)

df = df.dropna()

# print(df.head())

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

model = Sequential()
model.add(Dense(32, input_shape=(2,)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=10, verbose=1)

y_predicted = model.predict(x_test)

print(len(y_predicted))
print(len(y_test))

plt.plot(y_test)
plt.plot(y_predicted)
plt.show()