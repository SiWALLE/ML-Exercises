import numpy as np
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class Classifier(object):
    
    __name__ = 'base'
    training_set_X = []
    training_set_Y = []
    model = {}
    
    def __init__(self,Examples_X,Examples_Y):
        self.training_set_X = Examples_X
        self.training_set_Y = Examples_Y
        
    def sample_data(self,size_percent=0.1):
        Sampled_X = []
        Sampled_Y = []
        size = int(max(size_percent*len(self.training_set_X),1))
        randArray = np.random.randint(0,len(self.training_set_X)-1,size)
        for index in randArray:
            Sampled_X.append(self.training_set_X[index])
            Sampled_Y.append(self.training_set_Y[index])
        return np.array(Sampled_X), np.array(Sampled_Y)     
    
    def train__model(self,ifInit=False):
        # To be implemented in subclass
        pass
    
    def predict(self,x):
        # To be implemented in subclass
        pass
    
    def calculate_loss(self):
        # To be implemented in subclass
        pass
    
    def plot_decision_boundary(self):
        print self.__name__
        matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
        Figure()
        if(len(self.training_set_X[0])!=2): 
            print 'Not a 2D data set'
            return 
        x_min , x_max = self.training_set_X[:,0].min()-0.5 , self.training_set_X[:,0].max()+0.5
        y_min , y_max = self.training_set_X[:,1].min()-0.5 , self.training_set_X[:,1].max()+0.5
        h=0.01
        xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
        z = self.predict(np.c_[xx.ravel(),yy.ravel()])
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.get_cmap('Spectral'))
        plt.scatter(self.training_set_X[:, 0], self.training_set_X[:, 1], c=self.training_set_Y, cmap=plt.get_cmap('Spectral'))
        plt.show(block = False)