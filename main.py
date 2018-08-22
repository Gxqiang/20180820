from RoughFeatureSelection import *


data = np.load('BCW_data.npy')

x = data[:,:-1]; Y = data[:,-1]

y = []
for ii in Y:
	if ii==2:
		y.append(2.)
	elif ii==4:
		y.append(4.)
y = np.array(y)

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
sss=StratifiedShuffleSplit(n_splits=1,test_size=.4,random_state=0)
sss.get_n_splits(x,y)
for train_index,test_index in sss.split(x,y):
	x_train,x_test=x[train_index],x[test_index]
	y_train,y_test=y[train_index],y[test_index]
# fit a CART model to the data
fs = list(EFSA(x_train,y_train))
re = []
for i in sorted(fs):
	re.append(fs.index(i))
re = re[::-1]
print re[:3]
model = DecisionTreeClassifier()
# model = LinearSVC()
l = [0,1,2,4,5]
l = re[:3]
model.fit(x_train[:,l], y_train)
# print(model)
# make predictions
expected = y_test
predicted = model.predict(x_test[:,l])
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))