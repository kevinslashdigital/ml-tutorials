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
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats

class HousePredictor():

    def __init__(self, *args, **kwargs):
        pass

    def target_analysis(self,data, column):
        plt.subplots(figsize=(12,9))
        sns.distplot(data[column], fit=stats.norm)
        # Get the fitted parameters used by the function
        (mu, sigma) = stats.norm.fit(data[column])
        # plot with the distribution
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
        plt.ylabel('Frequency')
        #Probablity plot
        fig = plt.figure()
        stats.probplot(data[column], plot=plt)
        plt.show()
    
    def preprocessing(self):
        filename = 'dataset/house_price/kc_house_data.csv'
        df = pd.read_csv(filename)
        print(df.head())
        df = df[['bedrooms','bathrooms','sqft_living','condition','sqft_basement','yr_built','grade','price']]
        #we use log function which is in numpy
        df['price'] = np.log1p(df['price'])
        # self.target_analysis(df,'price')
        print('missing value', df.columns[df.isnull().any()])

        # return None, None, None, None

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
        # ax1 = sns.boxplot(x=df['bedrooms'])

        # print(df.head())
        print(df.shape)
        df = df[(np.abs(stats.zscore(df)) < 5).all(axis=1)]
        print(df.shape)
        # ax2 = sns.boxplot(x=df_new['bedrooms'])

      
        # col = ['bedrooms','bathrooms','sqft_living','sqft_basement','yr_built','grade','price']
        # sns.set(style='ticks')
        # sns.pairplot(df[col], size=3, kind='reg')

     

        # #Coralation plot
        # corr = df.corr()
        # plt.subplots(figsize=(20,9))
        # sns.heatmap(corr, annot=True)

        # top_feature = corr.index[abs(corr['price']>0.5)]
        # plt.subplots(figsize=(12, 8))
        # top_corr = df[top_feature].corr()
        # sns.heatmap(top_corr, annot=True)
        # # plt.show()
        # df = df[top_feature]
    
        # convert pf to numpy array
        dataset = df.values

        x_train, x_test, y_train, y_test = train_test_split(dataset[ :,:-1],  dataset[ :,-1], test_size=0.3, random_state=33)
        # x_test, y_test = dataset[ :,:-1], dataset[ :,-1]
        # x_train, y_train = dataset[ :,:-1], dataset[ :,-1]

        return x_train, y_train, x_test, y_test

    def train(self):
        train_data, train_label, test_data, test_label = self.preprocessing()
        # LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        reg = linear_model.LinearRegression() 
        # reg = RandomForestRegressor(n_estimators=1000)
        # reg = linear_model.Ridge(alpha = .5)
        # reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
        # reg = linear_model.Lasso(alpha = 0.7)
        # reg = linear_model.LassoLars(alpha=0.1)
        # reg = linear_model.BayesianRidge()
        # reg = linear_model.ElasticNet(random_state=4, selection='random')
        # reg = linear_model.SGDRegressor()
        # req = GradientBoostingRegressor(n_estimators=100, max_depth=4)
        # reg = Pipeline([('poly', PolynomialFeatures(degree=3)),
        #                 ('linear',linear_model.LinearRegression(fit_intercept=False))])
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
            print("Accuracy --> ", model.score(x_test, y_test)*100)
            # Explained variance score: 1 is perfect prediction
            print('Variance score: %.2f' % r2_score(y_test, y_predict))
            # The mean squared error
            print("Mean squared error: %.2f"% mean_squared_error(y_test, y_predict))
            # Plot outputs
            plt.plot(x_test[:,:1].flatten(), y_test,  color='black')
            plt.plot(x_test[:,:1].flatten(), y_predict, lw=2, alpha=0.3, label='ROC game (AUC = 0.02)')
    
            plt.show()
        else:
            print('prediction is ',y_predict)

