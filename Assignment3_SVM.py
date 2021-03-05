


from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

data=np.zeros((4, 2))
data=np.array([[1,1],[-1,-1],[1,0],[0,1]])
xmin, xmax = data[:, 0].min() - 1, data[:, 0].max() + 2
ymin, ymax = data[:, 1].min() - 1, data[:, 1].max() + 1
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Support vector machine with a hard margin')
xd=np.arange(-2, 3, 0.5)
yd = -1*xd + 1.5
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.scatter(data[:, 0], data[:, 1])
plt.grid(True)
plt.show()
#################################################################

iris = datasets.load_iris() #load data
X = iris.data[0:100, :2] # we only take the first two features
y = iris.target[0:100] #take the first 100 training examples 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0) 
#splitting data test and training data

# Fit the data to a logistic regression model
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

#Compute accuracy of Logistic Regression's on training and test sets
score = logisticRegr.score(x_train, y_train)
print(score)
score2 = logisticRegr.score(x_test, y_test)
print(score2)


# Retrieve the model parameters.
b = logisticRegr.intercept_[0]
print(b)
w = logisticRegr.coef_.T
print(w[0],w[1])
# Calculate the intercept and gradient of the decision boundary
c = -b/w[1]
a = -w[0]/w[1]


# Plot the training data and the classification with the decision boundary
xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
xd = np.array([xmin, xmax])
yd = a*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.scatter(X[:, 0], X[:, 1], c=y )
          
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Linear Binary Classification')

plt.show()

############################################################################

#Plot the test data

plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train
          )
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Linear Binary Classification on the Training Set')

plt.show()


######svm.SVC(kernel="linear"#########



##############################################################################

#Create a SVM Classifier
clf = svm.SVC(kernel="linear",C=1000) # Increased max_iter function to converge

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
y_pred_train= clf.predict(x_train)

#metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy on test examples, new imp:",metrics.accuracy_score(y_test, y_pred))
print("Accuracy on training examples , new imp:",metrics.accuracy_score(y_train, y_pred_train))

# Get the separating hyperplane
w = clf.coef_[0]
print("new w:",w)
print(clf.intercept_[0])

# Calculate the decision function 

support_vectors = clf.support_vectors_
print("support vectors:",support_vectors)

# Support vectors are margin away from hyperplane in direction perpendicular to hyperplane 

# Margin is 1/length(w)
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
print("value of margin:",margin)

a = -w[0] / w[1]
xd = np.linspace(xmin, xmax)
yd = a * xd - (clf.intercept_[0]) / w[1]

# Margin is sqrt(1+a^2) away vertically in 2-d
yd_down = yd - np.sqrt(1 + a ** 2) * margin
yd_up = yd + np.sqrt(1 + a ** 2) * margin

# Plotting the parallels to the separating hyperplane that pass through the support vectors
# Plotting the hyperplane
plt.plot(xd, yd, 'k-')
plt.plot(xd, yd_down,'k--')
plt.plot(xd, yd_up, 'k--')
plt.scatter(support_vectors[:, 0],support_vectors[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Classifier on Training Set kernel="linear"')

plt.show()
############################################################################

############################################################################

#Plot the test data

plt.plot(xd, yd, 'k-')
plt.plot(xd, yd_down,'k--')
plt.plot(xd, yd_up, 'k--')
plt.scatter(support_vectors[:, 0],support_vectors[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Classifier on Test Set kernel="linear"')

plt.show()

##############################################################################



##############################################################################

#Create a SVM Classifier
clf = svm.LinearSVC(C=1000000, max_iter=1000000) # Increased max_iter function to converge

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
y_pred_train= clf.predict(x_train)

#metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy on test examples:",metrics.accuracy_score(y_test, y_pred))
print("Accuracy on training examples:",metrics.accuracy_score(y_train, y_pred_train))

# Get the separating hyperplane
w = clf.coef_[0]
print(w)
print(clf.intercept_[0])

# Calculate the decision function 
decision_function = np.dot(x_train, clf.coef_[0]) + clf.intercept_[0]
support_vector_indices = np.where((2 * y_train - 1) * decision_function <= 1)[0]
support_vectors = x_train[support_vector_indices]
print("support vectors:",support_vectors)

# Support vectors are margin away from hyperplane in direction perpendicular to hyperplane 

# Margin is 1/length(w)
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
print("value of margin:",margin)

a = -w[0] / w[1]
xd = np.linspace(xmin, xmax)
yd = a * xd - (clf.intercept_[0]) / w[1]

# Margin is sqrt(1+a^2) away vertically in 2-d
yd_down = yd - np.sqrt(1 + a ** 2) * margin
yd_up = yd + np.sqrt(1 + a ** 2) * margin

# Plotting the parallels to the separating hyperplane that pass through the support vectors
# Plotting the hyperplane
plt.plot(xd, yd, 'k-')
plt.plot(xd, yd_down,'k--')
plt.plot(xd, yd_up, 'k--')
plt.scatter(support_vectors[:, 0],support_vectors[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Linear SVM Classifier on Training Set LinearSVC')

plt.show()
############################################################################

#Plot the test data

plt.plot(xd, yd, 'k-')
plt.plot(xd, yd_down,'k--')
plt.plot(xd, yd_up, 'k--')
plt.scatter(support_vectors[:, 0],support_vectors[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Linear SVM Classifier on Test Set')

plt.show()

##############################################################################
#######################SVC with RBF Kernel#################################


iris = datasets.load_iris() #load data
X = iris.data[0:150, :2] # we only take the first two features
y = iris.target[0:150] #take the first 100 training examples 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0) 

plt.scatter(X[:, 0], X[:, 1], c=y)

# Create a SVC classifier using an RBF kernel
svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=1)
# Train the classifier
svm.fit(x_train, y_train)

# Visualize the decision boundaries

markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(y))])

# plot the decision surface
x1_min, x1_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
x2_min, x2_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max),
                           np.arange(x2_min, x2_max))
Z = svm.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVC with RBF Kernel Classifier on Multi-Class Data')
for idx, cl in enumerate(np.unique(y_train)):
        plt.scatter(x=x_train[y_train == cl, 0], y=x_train[y_train == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
        
y_pred = svm.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))