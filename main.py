#svm    Support Vector machine 
import pandas as pd
from sklearn import datasets

cancer_dict = datasets.load_breast_cancer()
print(cancer_dict.keys())

data = pd.DataFrame(cancer_dict.data)
data.columns = cancer_dict.feature_names

data["isCancer"] = cancer_dict.target

print(data.info())
print(data.head())

y = data["isCancer"]

data.drop("isCancer",axis=1)
x = data

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size= 0.2, random_state=5)


from sklearn import svm 
model = svm.SVC(kernel="linear")

model.fit(xtrain,ytrain)

ypredict = model.predict(xtest)


import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,confusion_matrix

acuracy = accuracy_score(ytest,ypredict)
acuracy = round(acuracy*100,2)

print("Acuracy is : ",acuracy,"%")

matrix = confusion_matrix(ytest,ypredict)
sb.heatmap(matrix,annot=True,fmt="d")

plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Acuracy")

plt.show()

