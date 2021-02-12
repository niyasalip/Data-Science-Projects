import pandas as pd 
import pickle
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier



train = pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

data=pd.concat([train, test], axis=0)
print(data.head)
print(data.columns)


#splitting

y = data['fake']

x = data.drop('fake',axis=1 )

print('X')
print(x.head())


print('Y')
print(y.head())



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#Std Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)


print(x)

model.fit(x,y)

pickle.dump(model,open('model.pkl','wb'))