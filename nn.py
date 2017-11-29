import numpy as np
import pandas as pd

#loading data
X_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
X_test = data_test
Y_train = X_train.loc[:,["Survived"]]
X_train.drop(["Survived"], axis=1, inplace=True)
combined = pd.concat([X_train, X_test], axis = 0)

#processing the data
combined["Sex"] = combined["Sex"].map({"male":1, "female":0})
combined["Name"] = combined["Name"].map(lambda name: name.split(",")[1].split(".")[0].strip())
Pclass = pd.get_dummies(combined["Pclass"], prefix = "Pclass")
Embarked = pd.get_dummies(combined["Embarked"], prefix = "Embarked")
Title = pd.get_dummies(combined["Name"], prefix = "Title")
combined = pd.concat([combined, Pclass, Embarked, Title], axis = 1)
combined.drop(["PassengerId", "Pclass", "Name", "Embarked", "Ticket", "Cabin"], axis = 1, inplace = True)
combined["Age"].fillna(combined["Age"].median(), inplace = True)
combined["Fare"].fillna(combined["Fare"].median(), inplace = True)
del Embarked, Pclass, Title

X_train = combined.iloc[:891,:]
X_test = combined.iloc[891:,:]

X = X_train.loc[:,:].as_matrix()
X = X.T
X_test = X_test.loc[:,:].as_matrix()
X_test = X_test.T
Y = Y_train.loc[:,:].as_matrix()
Y = Y.T

n = X.shape[0]
m = X.shape[1]
m_test = X_test.shape[1]
W = np.zeros((n,1))
b = np.zeros((1,m))
alpha = 0.0005

for i in range(100000):
    Z = np.dot(W.T,X) + b
    A = 1/(1+np.exp(-Z))
    dZ = A-Y
    dW = (1/m)*np.dot(X,dZ.T)
    db = (1/m)*np.sum(dZ)
    W = W - alpha*dW
    b = b - alpha*db
b.resize((1,m_test))
Y_test = np.dot(W.T, X_test) + b
Y_test = Y_test.T

#making csv file
output = pd.DataFrame()
output["PassengerId"] = data_test["PassengerId"]
output["Survived"] = Y_test
for i in range(0, m_test):
    if (output["Survived"][i] < 0):
        output["Survived"][i] = 0
    else:
        output["Survived"][i] = 1
output["Survived"] = output["Survived"].astype(int) 
output[["PassengerId", "Survived"]].to_csv("gdTitanic.csv", index=False)
