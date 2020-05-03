import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.show()

train = pd.read_csv('titanic_train.csv')
train.drop('Cabin',axis=1,inplace=True)

def assumeage(col):
    pclass = col[0]
    age = col[1]
    
    if pd.isnull(age):
        if pclass == 1:
            return train[train['Pclass']==1]['Age'].mean()
        if pclass == 2:
            return train[train['Pclass']==2]['Age'].mean()
        if pclass == 3:
            return train[train['Pclass']==3]['Age'].mean()
    else:
        return age
       

train['Age'] = train[['Pclass','Age']].apply(assumeage,axis=1)
gender = pd.get_dummies(train['Sex'],drop_first = True)
embarked = pd.get_dummies(train['Embarked'],drop_first = True)

train = pd.concat([train,gender,embarked],axis=1)

    
#Using KNN
#from sklearn.preprocessing import StandardScaler

x = train[['Pclass','Age','SibSp','Parch','Fare','male','Q','S']]
y = train['Survived']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier

error_value = []
for i in range(1,51):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train,y_train)
    predictions = knn.predict(x_test)
    
    error_value.append(np.mean(predictions != y_test))
    
errorvaluesdf = pd.DataFrame(error_value,range(1,51))
#print(errorvaluesdf[errorvaluesdf[0]==errorvaluesdf[0].min()]) // 15

knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(x_train,y_train)
predictions_knn = knn.predict(x_test)

#Using Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train,y_train)
predictions_logreg = model.predict(x_test)

#Using a Decision Tree

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)
predictions_tree = dtree.predict(x_test)

#RFC
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 550)
rfc.fit(x_train,y_train)
predictions_rfc = rfc.predict(x_test)

#Accuracy
from sklearn.metrics import accuracy_score
knn_accuracy = accuracy_score(y_test,predictions_knn)    
logreg_accuracy = accuracy_score(y_test,predictions_logreg)
dtree_accuracy = accuracy_score(y_test,predictions_tree)
rfc_accuracy = accuracy_score(y_test,predictions_rfc)


print(f"Accuracy while using the KNN algorithm: {knn_accuracy}")
print(f"Accuracy while using the Logistic Reg. algorithm: {logreg_accuracy}")
print(f"Accuracy while using a decision tree algorithm: {dtree_accuracy}")
print(f"Accuracy while using a Random Forest algorithm: {rfc_accuracy}")

input('Press ENTER to exit')



    
    
    

