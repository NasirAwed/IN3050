#!/usr/bin/env python
# coding: utf-8

# ## IN3050/IN4050 Mandatory Assignment 2: Supervised Learning

#
#
# ### Name: Nasir Awed
#
# ### Username: nasiraa
#

# ### Intialization

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
import random
from collections import Counter
from sklearn.metrics import mean_squared_error


# In[6]:


from sklearn.datasets import make_blobs
X, t = make_blobs(n_samples=[400,800,400], centers=[[0,0],[1,2],[2,3]],
                  n_features=2, random_state=2019)

print(X.shape)


# In[7]:


indices = np.arange(X.shape[0])
random.seed(2020)
random.shuffle(indices)
indices[:10]


# In[8]:


X_train = X[indices[:800],:]
X_val = X[indices[800:1200],:]

X_test = X[indices[1200:],:]
t_train = t[indices[:800]]
t_val = t[indices[800:1200]]
t_test = t[indices[1200:]]


# Next, we will  make a second dataset by merging the two smaller classes in (X,t) and call the new set (X, t2). This will be a binary set.

# In[35]:


t2_train = t_train == 1
t2_train = t2_train.astype('int')
t2_val = (t_val == 1).astype('int')
t2_test = (t_test == 1).astype('int')


# Plot the two training sets.

# In[10]:



def show(X, y, marker='.'):
    labels = set(y)
    for lab in labels:
        plt.plot(X[y == lab][:, 1], X[y == lab][:, 0],
                 marker, label="class {}".format(lab))
    plt.legend()
show(X_train,t_train)


# In[11]:


show(X_train,t2_train)


# ## Binary classifiers

# ### Linear regression
# We see that that set (X, t2) is far from linearly separable, and we will explore how various classifiers are able to handle this. We start with linear regression. You may use the implementation from exercise set week07 or make your own. You should make one improvement. The implementation week07 runs for a set number of epochs. You provide the number of epochs with a parameter to the fit-method. However, you do not know what a reasonable number of epochs is. Add one more argument to the fit-method *diff* (with defualt value e.g. 0.001). The training should stop when the update is less than *diff*. The *diff* will save training time, but it may also be wise to not set it too small -- and not run training for too long -- to avoid overfitting.
#
# Train the classifier on (X_train, t2_train) and test for accuracy on (X_val, t2_val) for various values of *diff*. Choose what you think is optimal *diff*. Report accuracy and save it for later.

# In[579]:


def add_bias(X):
    # Put bias in position 0
    sh = X.shape
    if len(sh) == 1:
        #X is a vector
        return np.concatenate([np.array([1]), X])
    else:
        # X is a matrix
        m = sh[0]
        bias = np.ones((m,1)) # Makes a m*1 matrix of 1-s
        return np.concatenate([bias, X], axis  = 1)

class NumpyClassifier():
    """Common methods to all numpy classifiers --- if any"""

    def accuracy(self,X_test, y_test, **kwargs):
        pred = self.predict(X_test, **kwargs)
        if len(pred.shape) > 1:
            pred = pred[:,0]
        return sum(pred==y_test)/len(pred)

class NumpyLinRegClass(NumpyClassifier):

    def fit(self, X_train, y_train, gamma = 0.1, epochs=1000, diff=0.0001):
        """X_train is a Nxm matrix, N data points, m features
        t_train are the targets values for training data"""

        (k, m) = X_train.shape
        X_train = add_bias(X_train)
        last_result = np.zeros(m+1)

        self.theta = theta = np.zeros(m+1)

        stopp = 1
        done = True
        self.theta = theta = np.zeros(m+1)
        theta -= gamma / k *  X_train.T @ (X_train @ theta - t_train)
        mse = 2/m*X_train.T@(X_train@theta-t_train)
        count=0

        while done:
            #ending when update is less then diff or if number of epochs equal to count
            if sum(mse)**2 < diff or count == epochs :

                break
            else:
                theta -= gamma / k *  X_train.T @ (X_train @ theta - y_train)
                #y = theta
                count  += 1
                theta -= gamma / k *  X_train.T @ (X_train @ theta - y_train)
                #mean squre error to find out when to terminate
                mse = 2/m*X_train.T@(X_train@theta-t_train)






    def predict(self, x, threshold=0.5):
        z = add_bias(x)
        score = z @ self.theta
        return score>threshold


lin_cl = NumpyLinRegClass()

lin_cl.fit(X_train, t2_train)

linear_acc = lin_cl.accuracy(X_val, t2_val)

print(linear_acc)


# ### Logistic regression
# Do the same for logistic regression, i.e., add the *diff*, tune it, report accuracy, and store it for later.

# In[427]:


def logistic(x):
    return 1 / (1 + np.exp(-x))

class NumpyLogReg(NumpyClassifier):

    def fit(self, X_train, t_train, gamma = 0.1, epochs=100, diff=0.0001):


        (k, m) = X_train.shape
        X_train = add_bias(X_train)

        self.theta = theta = np.zeros(m+1)
        stopp = 1
        done = True
        self.theta = theta = np.zeros(m+1)
        count=0
        svar = 1
        svar2 = 1


        while done:
            count  += 1
            #ending if the number of iterations equals the epochs
            if count == epochs :
                break
            else:
                z = np.dot(X_train, theta)
                h = logistic(z)
                #computing update
                gradient = np.dot(X_train.T, (h - t_train)) / t_train.shape[0]
                theta -= gamma * gradient
                #if the update is less then diff
                if sum(theta)**2 < diff:
                    print(sum(theta))
                    break


        print("number of epochs ",count)

    def forward(self, X):
        return logistic(X @ self.theta)

    def score(self, x):
        z = add_bias(x)
        score = self.forward(z)
        return score

    def predict(self, x, threshold=0.5):
        z = add_bias(x)
        score = self.forward(z)
        # score = z @ self.theta
        return (score>threshold).astype('int')


# In[428]:



lr_cl = NumpyLogReg()
lr_cl.fit(X_train, t2_train, epochs=100, gamma = 0.1)

lin_accu = lr_cl.accuracy(X_val, t2_val)

print(lin_accu)


# ### *k*-nearest neighbors (*k*NN)
# We will now compare to the *k*-nearest neighbors classifier. You may use the implementation from the week05 exercise set. Beware, though, that we represented the data differently from what we do here, using Python lists instead of numpy arrays. You therefore have to either modify the representation of the data or the code a little.
#
# Train on (X_train, t2_train) and test on (X2_val, x2_val) for various values of *k*. Choose the best *k*, report accuracy and store for later.

# In[358]:


def distance_L2(a, b):
    "L2-distance using comprehension"
    s = sum((x - y) ** 2 for (x,y) in zip(a,b))
    return s ** 0.5

def majority(a):

    counts = Counter(a)
    return counts.most_common()[0][0]

class PyClassifier():
    """Common methods to all python classifiers --- if any"""

    def accuracy(self,X_test, y_test, **kwargs):
        """Calculate the accuracy of the classifier
        using the predict method"""
        predicted = [self.predict(a, **kwargs) for a in X_test]
        equal = len([(p, g) for (p,g) in zip(predicted, y_test) if p==g])
        return equal / len(y_test)


class PykNNClassifier(PyClassifier):
    """kNN classifier using pure python representations"""

    def __init__(self, k=3, dist=distance_L2):
        self.k = k
        self.dist = dist

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, a):
        X = self.X_train
        y = self.y_train
        distances = [(self.dist(a, b), b, c) for (b, c) in zip(X, y)]
        distances.sort()
        predictors = [c for (_,_,c) in distances[0: self.k]]

        return majority(predictors)


# In[359]:


knn_acc = []
Kk = [3,5,10]
for x in Kk:
    pyKNN = PykNNClassifier(k=x)
    X_train2 = X_train.copy()
    t2_train2 = t2_train.copy()
    pyKNN.fit(X_train2.tolist(),t2_train2.tolist())
    X_val2 = X_val.copy()
    t2_val2 = t2_val.copy()
    knn_acc.append(pyKNN.accuracy(X_val2.tolist(),t2_val2.tolist()))
knn_acc.sort()

print(knn_acc[-1])


# ### Simple perceptron
# Finally, run the simple perceptron (week05) on the same set, and report and store accuracy.

# In[17]:


class PyPerClassifier(PyClassifier):


    def fit(self, X_train, y_train, eta=0.1, epochs=1):
        """Train the self.weights on the training data eith learning
        rate eta, running epochs many epochs"""
        X_train = [[1]+list(x) for x in X_train] # Put bias in position 0
        self.dim = dim = len(X_train[0])
        self.weights = weights = [0 for _ in range(dim)]
        # Initialize all weights to 0. There are better ways!

        for e in range(epochs):
            for x, t in zip(X_train, y_train):
                y = int(self.forward(x)>0)
                for i in range(dim):
                    weights[i] += eta * (t - y) * x[i]

    def forward(self, x):
        """Calculate the score for the item x"""
        score = sum([self.weights[i]*x[i] for i in range(self.dim)])
        return score

    def predict(self, x):
        """Predict the value for the item x"""
        x = [1] + list(x)
        score = self.forward(x)
        return int(score > 0)


# In[18]:



perceptron = PyPerClassifier()
perceptron.fit(X_train2,t2_train2)
perc_acc= perceptron.accuracy(X_val2.tolist(),t2_val2.tolist())

print(perc_acc)


# ### Summary
# Report the accuracies for the four classifiers in a table.
#
# Write a couple of sentences where you comment on what you see. Are the results as you expected?

# In[265]:


from texttable import Texttable
tt = Texttable()
tt.add_rows([['classifiers', 'accuracies'], ['Linear regression', linear_acc], ['Logistic regression ', lin_accu],
           ['k-nearest neighbors',  knn_acc[-1]], ['Simple perceptron',  perc_acc]])
print(tt.draw())


# ## Multi-class classifiers
# We now turn to the task of classifying when there are more than two classes, and the task is to ascribe one class to each input. We will now use the set (X, t).

# ### *k*NN
# One of the classifiers can handle multiple classes without modifications: the *k*-nearest neighbors classifier. Train it on (X_train, t_train), test it on (X_val, t_val) for various values of *k*. Choose the one you find best and report the accuracy.

# In[20]:


knn_acc2 = []
Kk = [3,5,10]
for x in Kk:
    pyKNN2 = PykNNClassifier(k=x)
    X_train1 = X_train.copy()
    t1_train1 = t_train.copy()
    pyKNN2.fit(X_train1.tolist(),t1_train1.tolist())
    X_val1 = X_val.copy()
    t1_val1 = t_val.copy()
    knn_acc2.append(pyKNN2.accuracy(X_val1.tolist(),t1_val1.tolist()))
knn_acc2.sort()

print(knn_acc2[-1])


# ### Logistic regression "one-vs-rest"
#

# In[556]:



class NumpyLogReg(NumpyClassifier):



    def fit(self, X_train, t_train, gamma = 1, epochs=1000, diff=0.0001):


        (k, m) = X_train.shape
        X_train = add_bias(X_train)

        self.theta = theta = np.zeros(m+1)

        done = True
        #self.theta = theta = np.zeros(m+1)
        count=0

        while done:
            count  += 1
            #ending if the number of iterations exceeds the epochs
            if count == epochs :
                break
            else:
                z = np.dot(X_train, theta)
                h = logistic(z)
                #computing update
                gradient = np.dot(X_train.T, (h - t_train)) / t_train.shape[0]
                theta -= gamma * gradient


                #if the update is less then diff
                if sum(theta)**2 < diff:
                    print("number of epocs", count)
                    break



    def forward(self, X):
        return logistic(X @ self.theta)

    def score(self, x):
        z = add_bias(x)
        score = self.forward(z)
        return score

    def predict(self, x, threshold=0.5):
        z = add_bias(x)
        score = self.forward(z)
        # score = z @ self.theta
        return (score>threshold).astype('int')


# In[22]:


multi_log = NumpyLogReg()
en = t_train.copy()
to = t_train.copy()
tre = t_train.copy()

en_val = t_val.copy()
to_val = t_val.copy()
tre_val = t_val.copy()

# here i make 3 different classifires for the 3 different classes

for x in range(len(en)):
    if en[x] != 0:
        en[x]= 0
    else:
        en[x]=1

    if to[x] != 1:
        to[x] = 0
    else:
        to[x]=1

    if tre[x] != 2:
        tre[x] = 0
    else:
        tre[x]=1

for x in range(len(t_val)):
    if en_val[x] != 0:
        en_val[x]= 0
    else:
        en_val[x]=1

    if to_val[x] != 1:
        to_val[x] = 0
    else:
        to_val[x]=1

    if tre[x] != 2:
        tre[x] = 0
    else:
        tre[x]=1

classes = []
classes.append((en,en_val))
classes.append((to,to_val))
classes.append((tre,tre_val))

accuracies = []

for x in range(len(classes)):
    multi_log.fit(X_train, classes[x][0], epochs=1000, gamma = 0.01)
    multi_accu = multi_log.accuracy(X_val, classes[x][1])
    accuracies.append(multi_accu)


print("the best",max(accuracies))
print("with class",accuracies.index(max(accuracies)))
#class 0 at index 0, class 1 at index 1 and class 2 at index 2
print(accuracies)



#
# Discuss the results in a couple of sentences, addressing questions like
#
# - How do the two classfiers compare?
#
#
# - How do the results on the three-class classification task compare to the results on the binary task?
# - What do you think are the reasons for the differences?
#
#

# # Answer
# - it seems like there is a big differense between the classes. where class 0 is the clear winner
# - number of epocs 1
# - the best 0.8525
# - with class 0
# - [0.8525, 0.5875, 0.285]
#
# - but class 0 has a much better result then the binary task
#
# - I think the reason for the difference is that the classifirer is good at finding the features of the class 0 and not 1 and 2.

# ## Adding non-linear features

# We are returning to the binary classifier and the set (X, t2). As we see, some of the classifiers are not doing too well on the (X, t2) set. It is easy to see from the plot that this data set is not well suited for linear classifiers. There are several possible options for trying to learn on such a set. One is to construct new features from the original features to get better discriminants. This works e.g. on the XOR-problem. The current classifiers use two features: $x_1$ and $x_2$ (and a bias term $x_0$). Try to add three additional features of the form ${x_1}^2$, ${x_2}^2$, $x_1*x_2$ to the original features and see what the accuracies are now. Compare to the results for the original features in a 4x2 table.
#
# Explain in a couple of sentences what effect the non-linear features have on the various classifiers. (By the way, some of the classifiers could probably achieve better results if we scaled the data, but we postpone scaling to part 2 of the assignment.)

# In[304]:




multi_Xtrain = np.zeros((800,1))
multi_Xval = np.zeros((400,1))
# creating a array of features on the form of  𝑥1∗𝑥2 for X val and X train

for z in range(len(X_train)):
    for y in range(len(X_train[z])-1):
        multi_Xtrain[z][y] =  X_train[z][y]* X_train[z][y+1]

for z in range(len(X_val)):
    for y in range(len(X_val[z])-1):
        multi_Xval[z][y] =  X_val[z][y]* X_val[z][y+1]

# concatenate the array of 𝑥1´2, x2´2 and the normal X train
x2 =  X_train**2
nyt = np.concatenate((X_train, x2), axis=1)
# the x1*x2 array
nyt2 = np.concatenate((nyt, multi_Xtrain), axis=1)

# concatenate the array of 𝑥1´2, x2´2 and the normal X val
x2val = X_val**2
nutt = np.concatenate((X_val, x2val), axis=1)
# same for X val, x1*x2 array
nutt2 = np.concatenate((nutt, multi_Xval), axis=1)



linear = NumpyLinRegClass()
linear.fit(nyt2, t2_train)
linear_result = linear.accuracy(nutt2, t2_val)
#print(linear_result)



nyp = NumpyLinRegClass()
nyp.fit(nyt2, t2_train, epochs=1000, gamma = 0.001)

result_nyp = nyp.accuracy(nutt2, t2_val)
#print(result_nyp)

pp = PyPerClassifier()
pp.fit(nyt2,t2_train2)
perc_acc= perceptron.accuracy(nutt2.tolist(),t2_val2.tolist())

#print(perc_acc)


knn_acc3 = []
Kk = [3,5,10]
for x in Kk:
    pyKNN2 = PykNNClassifier(k=x)
    X_train2 = nyt2.copy()
    t1_train3 = t2_train.copy()
    pyKNN2.fit(X_train2.tolist(),t1_train3.tolist())
    X_val2 = nutt2.copy()
    t1_val2 = t2_val.copy()
    knn_acc3.append(pyKNN2.accuracy(X_val2.tolist(),t2_val2.tolist()))
knn_acc3.sort()

#print(knn_acc2[-1])




tt = Texttable()
tt.add_rows([['classifiers', 'old accuracies', 'new accuracies'], ['Linear regression', linear_acc, linear_result],
             ['Logistic regression ', lin_accu, result_nyp],
           ['k-nearest neighbors',  knn_acc[-1],knn_acc2[-1]], ['Simple perceptron',  perc_acc, perc_acc]])
print(tt.draw())

"""
I can se that the logistic regression improved. as the the model gets more information from the raw data.
The linear model is worse, i think its because the added feauters did not correlate well with the other information.
or maybe because of multicollinearity.

k-nearest neighbors, Simple perceptron has no change.

"""


# # Part II
# ## Multi-layer neural networks
# We will now implement the Multi-layer feed forward network (MLP, Marsland sec. 4.2.1). We will do it in two steps. In the first step, we will work concretely with the dataset (X, t). We will initailize the network and run a first round of training, i.e. one pass throught the algorithm at p. 78 in Marsland.
#
# In the second step, we will turn this code into a more general classifier. We can train and test this on (X, t), but also on other datasets.
#
# First of all, you should scale the X.

# In[142]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train_scaled = X_scaled[indices[:800],:]
X_val_scaled = X_scaled[indices[800:1200],:]

X_test_scaled = X_scaled[indices[1200:],:]
t_train_scaled = t[indices[:800]]
t_val_scaled = t[indices[800:1200]]
t_test_scaled = t[indices[1200:]]


# ## Step1: One round of training

# ### Intializing
# We will only use one hidden layer. The number of nodes in the hidden layer will be a hyper-parameter provided by the user; let's call it *dim_hidden*. (*dim_hidden* is called *M* by Marsland.) Initially, we will set it to 6. This is a hyper-parameter where other values may give better results, and the hyper-parameter could be tuned.
#
# Another hyper-parameter set by the user is the learning rate. We set the initial value to 0.01, but also this may need tuning.

# In[41]:


eta = 0.01 #Learning rate
dim_hidden = 6


# We assume that the input *X_train* (after scaling) is a matrix of dimension *P x dim_in*, where *P* is the number of training instances, and *dim_in* is the number of features in the training instances (*L* in Marsland). Hence we can read *dim_in* off from *X_train*. Similarly, we can read *dim_out* off from *y_train*. Beware that *y_train* must be given the form of *P x dim_out* at some point, cf. the "one-vs-all" exercise above.

# In[201]:


dim_in =  len(X_train_scaled[0])  # Calculate the correct value from the input data

dim_out = len(set(t_train)) # Calculate the correct value from the input data
print(dim_in)
print(dim_out)


# We need two sets of weights: weights1 between the input and the hidden layer, and weights2, between the hidden layer and the output. Make the weight matrices and initialize them to small random numbers. Make sure that you take the bias terms into consideration and get the correct dimensions.

# In[557]:



weights1 = np.random.random((2,6))
weights2 = np.random.random((6,3))


# ### Forwards phase
# We will run the first step in the training, and start with the forward phase. Calculate the activations after the hidden layer and after the output layer. We will follow Marsland and use the logistic (sigmoid) activation function in both layers. Inspect whether the results seem reasonable with respect to format and values.

# In[558]:





def log2(y):
        return y*(1-y)

def forwardd(Xx):

    hidden_activations = []
    output_activations = []
    #print(Xx.shape,weights1.shape )
    h = np.dot(Xx, weights1)
    hidden_activations = logistic(h)
    z = np.dot(hidden_activations, weights2)
    output_activations = logistic(z)

    return hidden_activations, output_activations


# ### Backwards phase
# Calculate the delta terms at the output. We assume, like Marsland, that we use sums of squared errors. (This amounts to the same as using the mean square error).

# In[559]:


def backward( Xx, y,weights1,weights2 ):
    hidden, output = forwardd(Xx)
    output_error = y - output
    output_delta = output_error*log2(output)

    hidden_error = output_delta.dot(weights2.T)
    hidden_delta = hidden_error * log2(hidden)

    weights1 += Xx.T.dot(hidden_delta)
    weights2 += hidden_delta.T.dot(output_delta)


# ##  Step 2: A Multi-layer neural network classifier

# You want to train and test a classifier on (X, t). You could have put some parts of the code in the last step into a loop and run it through some iterations. But instead of copying code for every network we want to train, we will build a general Multi-layer neural network classfier as a class. This class will have some of the same structure as the classifiers we made for linear and logistic regression. The task consists mainly in copying in parts from what you did in step 1 into the template below. Remember to add the *self*- prefix where needed, and be careful in your use of variable names.

# In[584]:


class MNNClassifier():
    """A multi-layer neural network with one hidden layer"""

    def __init__(self,eta = 0.0021, dim_hidden = 6):
        """Intialize the hyperparameters"""
        self.eta = eta
        self.dim_hidden = dim_hidden

        # Should you put additional code here?

    def fit(self, X_train, t_train, epochs = 100):
        """Intialize the weights. Train *epochs* many epochs."""

        # Initilaization
        # Fill in code for initalization
        (k, m) = X_train.shape
        #X_train = add_bias(X_train)

        self.theta = theta = np.zeros(m+1)
        t_train = t_train.T
        X_scale = scaler.fit_transform(X_train)


        for e in range(epochs):
            # Run one epoch of forward-backward
            #Fill in the code
            #hidden_activations,output_activations=forward(X_train)
            backward(X_scale, new_t,weights1,weights2)
            #print(output_activations[0])
            theta -= 0.01 / k *  output_activations.T @ (output_activations @ theta - t_train)




    def forward(self, X):
        """Perform one forward step.
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""

        hidden_activations,output_activations = forwardd(X)
        return  hidden_activations,output_activations




        """Calculate the accuracy of the classifier on the pair (X_test, t_test)
        Return the accuracy"""
    def accuracy(self,X_test, y_test, **kwargs):
        pred = self.predict(X_test, **kwargs)
        if len(pred.shape) > 1:
            pred = pred[:,0]
        return sum(pred==y_test)/len(pred)



    def predict(self, x, threshold=0.5):
        z = add_bias(x)
        score = z @ self.theta
        return score>threshold



t_train = t_train.T
X_scale = scaler.fit_transform(X_train)
X_vall= scaler.transform(X_val)
new_t = np.array([t_train]).T
t_vall = np.array([t_val]).T



tester = MNNClassifier()
tester.fit(X_train_scaled, t_train_scaled)
X_vall= scaler.transform(X_val)
acc = tester.accuracy(X_vall,t_val_scaled)
print("acc",int(acc*100),"%")
