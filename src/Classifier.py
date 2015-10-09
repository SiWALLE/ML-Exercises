import numpy as np

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