from sklearn import linear_model
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

class HousePredictor():

    def __init__(self, *args, **kwargs):
        pass
    
    def preprocessing(self):
        filename = 'dataset/house_price/train_house_price_kaggle.csv'

        df = pd.read_csv(filename)
        print(df.head())
        df = df[['bedrooms','bathrooms','sqft_living','sqft_lot','sqft_above','condition','grade','price']]
    
        # convert city to categorical feature
        cons_binarizer = LabelBinarizer()
        cons_one_hot = cons_binarizer.fit_transform(df['condition'])
        pd_cons = pd.DataFrame(cons_one_hot, columns=cons_binarizer.classes_)
        
        # merge with one_hot
        df = df.merge(pd_cons, left_index=True,right_index=True, how='outer')
        
        # move price column to the last
        cols_at_end = ['price']
        df = df[[c for c in df if c not in cols_at_end] + [c for c in cols_at_end if c in df]]

        del df['condition']
        print(df.head())
        
        # convert pf to numpy array
        dataset = df.values

        x_train, x_test, y_train, y_test = train_test_split(dataset[ :,:-1],  dataset[ :,-1], test_size=0.3, random_state=33)
        # x_test, y_test = dataset[ :,:-1], dataset[ :,-1]
        # x_train, y_train = dataset[ :,:-1], dataset[ :,-1]

        return x_train, y_train, x_test, y_test

    def train(self):
        train_data, train_label, test_data, test_label = self.preprocessing()
        # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        # reg = linear_model.LinearRegression()
        # reg = linear_model.Ridge(alpha = .5)
        reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
        # reg = linear_model.Lasso(alpha = 0.7)
        # reg = linear_model.LassoLars(alpha=0.1)
        # reg = linear_model.BayesianRidge()
        # reg = linear_model.ElasticNet(random_state=4, selection='random')
        # reg = linear_model.SGDRegressor()
        reg = Pipeline([('poly', PolynomialFeatures(degree=3)),
                        ('linear',linear_model.LinearRegression(fit_intercept=False))])
        reg.fit(train_data,train_label)
        # reg.named_steps['linear'].coef_
        # print('coef_ = {} , intercept_ = {}'.format(reg.coef_, reg.intercept_))
        self.predict(reg,test_data,test_label)
        # print(reg.coef_)
        return reg

    def predict(self, model,x_test,y_test=None):
        # Make predictions using the testing set
        y_predict = model.predict(x_test)
        y_test = y_test.astype(float)
    
        if y_test is not None: 
            print(y_test)
            print(y_predict)
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % r2_score(y_test, y_predict))
            # The mean squared error
            print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predict))
            # Plot outputs
            plt.plot(x_test[:,:1].flatten(), y_test,  color='black')
            plt.plot(x_test[:,:1].flatten(), y_predict, lw=2, alpha=0.3, label='ROC game (AUC = 0.02)')
    
            # plt.show()
        else:
            print('prediction is ',y_predict)

