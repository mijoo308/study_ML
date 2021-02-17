
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

'''
sklearn
'''

#--- parameter ------
batch = 50  # batch size
max_epoch = 100

alpha = 0.1

#------------------

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))



#----------------------------------------
SGD_model = linear_model.SGDRegressor(alpha=alpha, average=True, max_iter=max_epoch, early_stopping=True,)
SGD_model.fit(X_train, y_train)


#-----------------------------

for i in range(max_epoch):
    for batch in get_batches(X_train,y_train,batch, weight, bias):
        w_grad = evalutate_gradient()
        b_grad = evaluate_gradient()

#----------------------------

# def SGD(data,)

