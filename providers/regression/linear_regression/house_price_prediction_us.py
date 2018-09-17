from sklearn import linear_model
import numpy as np
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

        # convert city to categorical feature
        # Create NumPy array
        cities = ['Normaltown', 'Hipstertown','Skidrow']
        # Create MultiLabelBinarizer object
        city_one_hot = LabelBinarizer()
        city_one_hot = city_one_hot.fit_transform(cities)
        print(cities)
     
        temp_x, temp_y = raw_data[ :,:-1],raw_data[ :,-1]
        dataset = np.concatenate((temp_x , city_one_hot), axis=0)
        dataset = np.concatenate((dataset , temp_y), axis=0)
        print('dataset',dataset)

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

