from sklearn import linear_model
import numpy
import warnings
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# from numpy import genfromtxt

class HousePredictor():

    def __init__(self, *args, **kwargs):
        pass
    
    def preprocessing(self):
        path_train = 'dataset/house_price/train_data.csv'
        path_test_x = 'dataset/house_price/test_predict.csv'
        path_test_y = 'dataset/house_price/sampleSubmission.csv'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_train = numpy.loadtxt(open(path_train, "r"), delimiter=",", skiprows=1)
            data_test_x = numpy.loadtxt(open(path_test_x, "r"), delimiter=",", skiprows=1)
            data_test_y = numpy.loadtxt(open(path_test_y, "r"), delimiter=",", skiprows=1)
        # result = genfromtxt(path, delimiter=",", skip_header=1)
        x_train ,y_train = data_train[:, :-1], data_train[:, -1]
        x_test_index = data_test_x[:2500,0: 1]
        x_test_index = x_test_index.flatten()
        print(x_test_index)
        data_test_y = numpy.take(data_test_y, numpy.array(x_test_index, dtype=numpy.int64))
        print("data_test_y", data_test_y)
        x_test, y_test = data_test_x[:2500,1:], data_test_y
   
        return x_train, y_train, x_test, y_test

    def train(self):
        train_data, train_label, test_data, test_label = self.preprocessing()
        # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        reg = linear_model.LinearRegression()
        reg.fit(train_data,train_label)
        self.predict(reg,test_data,test_label)
        # print(reg.coef_)
        return reg

    def predict(self, model,x_test,y_test=None):
        # Make predictions using the testing set
        y_predict = model.predict(x_test)
        if y_test is not None: 
            print(y_predict)
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % r2_score(y_test, y_predict))
            # The mean squared error
            print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predict))

            # Plot outputs
            plt.scatter(x_test[:,:1], y_test,  color='black')
            plt.plot(x_test[:,:1], y_predict, color='blue', linewidth=3)

            plt.xticks(())
            plt.yticks(())

            plt.show()
        else:
            print('prediction is ',y_predict)

