from sklearn import linear_model

from Classifier import Classifier


class LR_Classifier(Classifier):
    
    clf = linear_model.LogisticRegressionCV()
    
    """
    1\  h_dim is the nodes number of hidden layers
    2\  output_dim is the dimension of output layer, 
        often set as the classes of data
    """
    def __init__(self,Examples_X,Examples_Y,h_dim=1,output_dim=1):
        self.training_set_X = Examples_X
        self.training_set_Y = Examples_Y
        self.__name__ = 'Logistic Regression Classifier'
        
    def calculate_loss(self):
        pass
    
    def train_model(self):
        self.clf.fit(self.training_set_X,self.training_set_Y)
                
    def predict(self,x):
        return self.clf.predict(x)