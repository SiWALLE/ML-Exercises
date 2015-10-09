from sklearn import datasets
import numpy as np
from NN_Classifier import NN_Classifier
from LR_Classifier import LR_Classifier
import matplotlib
import matplotlib.pyplot as plt

def plot_decision_boundary(clf):
    print clf.__name__
    matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
    if(len(clf.training_set_X[0])!=2): 
        print 'Not a 2D data set'
        return 
    x_min , x_max = clf.training_set_X[:,0].min()-0.5 , clf.training_set_X[:,0].max()+0.5
    y_min , y_max = clf.training_set_X[:,1].min()-0.5 , clf.training_set_X[:,1].max()+0.5
    h=0.01
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.get_cmap('Spectral'))
    plt.scatter(clf.training_set_X[:, 0], clf.training_set_X[:, 1], c=clf.training_set_Y, cmap=plt.get_cmap('Spectral'))

np.random.seed(0)
All_Exam_X, All_Exam_Y = datasets.make_moons(200, noise=0.2)

plt.figure(1)
plt.subplot(211)
lr = LR_Classifier(All_Exam_X,All_Exam_Y)
lr.train_model()
plot_decision_boundary(lr)

plt.subplot(212)
nn = NN_Classifier(All_Exam_X,All_Exam_Y,h_dim=3,output_dim=2)
nn.train_model(20000,ifSample=False,print_loss=True)
plot_decision_boundary(nn)

plt.show()
