# Author yxp5

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 

df1 = pd.read_excel('diabetes data.xlsx', usecols='B,E,F,G,H,AN')
print(df1)

X = df1.drop('A1C', axis=1)
Y = df1['A1C']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=37)

model = LinearRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

# model evaluation 
print( 
  'mean_squared_error : ', mean_squared_error(Y_test, predictions)) 
print( 
  'mean_absolute_error : ', mean_absolute_error(Y_test, predictions)) 


from sklearn.svm import SVC  
clf = SVC(kernel='linear')
Y_train_bool = Y_train.apply(lambda x: x >= 5.7)
Y_test_bool = Y_test.apply(lambda x: x >= 5.7)
 
# fitting x samples and y classes 
clf.fit(X_train, Y_train_bool)
Y_pred = clf.predict(X_test)

diff = 0
for i, b in enumerate(Y_test_bool):
    diff += Y_pred[i] ^ b

length = len(X_test)
accuracy = round((length - diff) / length * 100, 2)
print(f"Accuracy is {accuracy}%")

me = {'age': 22, 'hight': 1.78, 'weight': 70, 'BMI': 22.1, 'IBW': 73}
a = pd.Series(me)
a = a.values.reshape(1, len(me))
print(clf.predict(a)) # I h8 my l1f3













