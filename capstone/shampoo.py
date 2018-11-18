#
# from pandas import read_csv
# from pandas import datetime
# from matplotlib import pyplot
# # load dataset
# def parser(x):
# 	return datetime.strptime('190'+x, '%Y-%m')
# series = read_csv('data/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# # summarize first few rows
# print(series.head())
# # line plot
# series.plot()
# pyplot.show()


from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np

X = np.random.random((10,3))
y = np.random.random((10,2))
X2 = np.random.random((7,3))

knn = KNeighborsRegressor()
regr = MultiOutputRegressor(knn)

regr.fit(X,y)
aa = regr.predict(X2)
aa
