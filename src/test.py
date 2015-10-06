import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from NN_Classifier import NN_classifier


matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

np.random.seed(0)
All_Exam_X, All_Exam_Y = datasets.make_moons(500, noise=0.2)

def sample_data(Examples_X,Examples_Y,size_percent = 0.005):
    Sampled_X = []
    Sampled_Y = []
    size = int(max(size_percent*len(Examples_X),1))
    randArray = np.random.randint(0,len(Examples_X)-1,size)
    for index in randArray:
        Sampled_X.append(Examples_X[index])
        Sampled_Y.append(Examples_Y[index])
    return np.array(Sampled_X), np.array(Sampled_Y)
    
plt.scatter(All_Exam_X[:, 0],All_Exam_X[:, 1], s=40, c=All_Exam_Y, cmap=plt.get_cmap('Spectral'))

clf = linear_model.LogisticRegressionCV()
clf.fit(All_Exam_X, All_Exam_Y)

def plot_decision_boundary(pred_func):
    x_min, x_max = All_Exam_X[:, 0].min() - 0.5, All_Exam_X[:, 0].max() + 0.5
    y_min, y_max = All_Exam_X[:, 1].min() - 0.5, All_Exam_X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), \
                        np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.get_cmap('Spectral'))
    plt.scatter(All_Exam_X[:, 0], All_Exam_X[:, 1], c=All_Exam_Y, cmap=plt.get_cmap('Spectral'))

# plot_decision_boundary(lambda x:clf.predict(x))
# plt.title("LogisticRegression")


nn = NN_classifier(All_Exam_X,All_Exam_Y,h_dim=3,output_dim=2)
nn.train_model(20000,print_loss=True)
nn.plot_decision_boundary()
