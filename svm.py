import tensorflow as tf
import tflearn
import numpy as np
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
import pickle
from sklearn.model_selection import cross_val_score
from skimage import feature
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
#Load Images from Paper
X = []
hog=[]
for i in range(1, 401):
    image = cv2.imread('Dataset/paper/paper_' + str(i) + '.png',0)
    #X.append(image.reshape(300, 300, 1))
    # extract Histogram of Oriented Gradients from the logo
    H = feature.hog(image, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    #X.append(image.flatten())
    X.append(H)


#Load Images From Rock
for i in range(1, 401):
    image = cv2.imread('Dataset/rock/rock_' + str(i) + '.png',0)
    H = feature.hog(image, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    #X.append(image.flatten())
    X.append(H)


for i in range(1, 401):
    image = cv2.imread('Dataset/sci/sci_' + str(i) + '.png',0)
    H = feature.hog(image, orientations=9, pixels_per_cell=(10, 10),
		cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    #X.append(image.flatten())
    X.append(H)

# Create OutputVector

Y = []
for i in range(0, 400):
    Y.append([1,0,0])

for i in range(0, 400):
    Y.append([0,1,0])

for i in range(0, 400):
    Y.append([0,0,1])


X = np.array(X)
Y = np.array(Y)
print X.shape
print Y.shape
#n_samples = len(X)
#X = X.reshape((n_samples, -1))
# Shuffle Training Data

X,Y = shuffle(X,Y, random_state=0)
Y = np.argmax(Y, axis=1)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.5,random_state=7,stratify=Y)


#'C': np.logspace(-3,3,50)
c=[0.01,0.1,1,10,100]
tuned_parameters = [{'kernel':  ['rbf','poly'], 'C': c}]
clf = GridSearchCV(svm.SVC(), tuned_parameters)
clf.fit(x_train, y_train)

sc=clf.best_score_
ker=clf.best_estimator_.kernel
bestc=clf.best_estimator_.C
print "Best Score according to grid search: ", sc*100,"\nBest kernel : ",ker, "\nBest C : ", bestc

mod=svm.SVC(gamma='auto',kernel=ker,C=bestc,verbose=0)
mod.fit(x_train,y_train)
yp=mod.predict(x_test)
accu=accuracy_score(y_test,yp)
print "New accuracy with optimal parameters : ",accu


"""
# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001,kernel='poly',C=10)
#scores = cross_val_score(clf, X, Y, cv=10)
# We learn the digits on the first half of the digits
clf.fit(x_train,y_train)
predicted = clf.predict(x_test)

print("Classification report - ",accuracy_score(y_test, predicted))
"""
#print predicted

# save the model to disk
filename = 'svm_model_2.sav'
joblib.dump(clf, filename)







