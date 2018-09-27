from sklearn import linear_model
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# https://engineering.hexacta.com/pandas-by-example-columns-547696ff78dd
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

        
        df = pd.DataFrame(data=raw_data[:,:])
        pd_data = pd.get_dummies(df, columns=[2])
        
        # reorder column
        pd_data = pd_data[[0, 1, '2_Hipstertown', '2_Normaltown','2_Skidrow',3]]

        price_col = pd_data.pop(3)
        pd_data = pd.concat([pd_data, price_col.rename("price")], axis=1) #{0/’index’, 1/’columns’}, default 0

        # # convert city to categorical feature
        # cities = df[2].tolist()
        # label_binarizer = LabelBinarizer()
        # city_one_hot = label_binarizer.fit_transform(cities)
        # pd_cons = pd.DataFrame(city_one_hot, columns=label_binarizer.classes_)
        # # merge with one_hot
        # df = df.merge(pd_cons, left_index=True,right_index=True, how='outer')
        # # move price column to the last
        # cols_at_end = [3]
        # df = df[[c for c in df if c not in cols_at_end] + [c for c in cols_at_end if c in df]]
        # del df[2]
        # dataset = df.values 
        
        # convert dataframe to numpy
        dataset = pd_data.values  
        print(pd_data.head())

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
        reg = linear_model.LinearRegression()
        # reg = linear_model.Ridge(alpha = .01)
        # reg = linear_model.Lasso(alpha = 0.7)
        # reg = linear_model.LassoLars(alpha=.1)
        # reg = linear_model.BayesianRidge()
        # reg = linear_model.ElasticNet(alpha = 0.02)
        # reg = linear_model.SGDRegressor()
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
            self.show_graph_bar(x_test,y_test,y_predict)
          
        else:
            print('prediction is ',y_predict)

    def show_graph(self,x_test,y_test,y_predict):
        plt.plot(x_test[:,:1].flatten(), y_test,  color='black')
        plt.plot(x_test[:,:1].flatten(),y_predict, lw=2, alpha=0.3, label='ROC game (AUC = 0.02)')
    
        plt.show()
    
    def show_graph_bar(self,x_test,y_test,y_predict):

        ind = np.arange(len(y_test))  # the x locations for the groups
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind - width/2, y_test, width,
                        color='SkyBlue', label='actual')
        rects2 = ax.bar(ind + width/2, y_predict, width,
                        color='IndianRed', label='predict')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('price')
        ax.set_title('price of the house')
        # ax.set_xticks(ind)
        # ax.set_xticklabels(y_test)
        ax.legend()

        self.autolabel(ax, rects1, "left")
        self.autolabel(ax, rects2, "right")
        plt.show()

    def autolabel(self,ax, rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')
