from sklearn import linear_model
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

class HousePredictor():

    def __init__(self, *args, **kwargs):
        pass
    
    def preprocessing(self):
        raw_data =  np.array([
            [3, 2000, 'Normaltown', 250000],
            [2, 800, 'Hipstertown', 300000],
            [2, 850, 'Normaltown', 150000],
            [1, 550, 'Normaltown', 78000],
            [4, 2000, 'Skidrow', 150000]
        ])

        pd_data = pd.DataFrame(data=raw_data[:,:])
        pd_data = pd.get_dummies(pd_data, columns=[2])
        pd_data = pd_data[[0, 1, '2_Hipstertown', '2_Normaltown','2_Skidrow',3]]
        dataset = pd_data.values

        # convert city to categorical feature
        # cities = pd_data[2].tolist()
        # city_one_hot = LabelBinarizer()
        # city_one_hot = city_one_hot.fit_transform(cities)
        # print(cities,city_one_hot)

        # dataset =  np.array([
        #     [3, 2000, 0 ,1, 0, 250000],
        #     [2, 800, 1 ,0, 0, 300000],
        #     [2, 850, 0 ,1, 0, 150000],
        #     [1, 550, 0 ,1, 0, 78000],
        #     [4, 2000, 0 ,0, 1, 150000]
        # ])

        x_test, y_test = dataset[ :,:-1], dataset[ :,-1]
        x_train, y_train = dataset[ :,:-1], dataset[ :,-1]

        return x_train, y_train, x_test, y_test

    def train(self):
        train_data, train_label, test_data, test_label = self.preprocessing()
        # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        # reg = linear_model.LinearRegression()
        # reg = linear_model.Ridge(alpha = .01)
        # reg = linear_model.Lasso(alpha = 0.7)
        # reg = linear_model.LassoLars(alpha=.1)
        # reg = linear_model.BayesianRidge()
        # reg = linear_model.ElasticNet(alpha = 0.02)
        reg = linear_model.SGDRegressor()
        reg.fit(train_data,train_label)
        print('coef_ = {} , intercept_ = {}'.format(reg.coef_, reg.intercept_))
        self.predict(reg,test_data,test_label)
        # print(reg.coef_)
        return reg

    def predict(self, model,x_test,y_test=None):
        # Make predictions using the testing set
        y_predict = model.predict(x_test)
        y_test = y_test.astype(float)
        if y_test is not None: 
            print(y_test, y_predict)
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % r2_score(y_test, y_predict))
            # The mean squared error
            print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predict))
            # Plot outputs
            plt.plot(x_test[:,:1].flatten(), y_test,  color='black')
            plt.plot(x_test[:,:1].flatten(),y_predict, lw=2, alpha=0.3, label='ROC game (AUC = 0.02)')
       
            plt.show()
        else:
            print('prediction is ',y_predict)

