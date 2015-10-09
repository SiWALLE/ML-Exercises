import numpy as np

from Classifier import Classifier

class NN_Classifier(Classifier):
    
    hidden_dim = 1
    output_dim = 1
    reg_lambda = 0.01
    epsilon = 0.01
    
    """
    1\  h_dim is the nodes number of hidden layers
    2\  output_dim is the dimension of output layer, 
        often set as the classes of data
    """
    def __init__(self,Examples_X,Examples_Y,h_dim=1,output_dim=1):
        self.training_set_X = Examples_X
        self.training_set_Y = Examples_Y
        self.__name__ = 'Neural Network Classifier'
        self.hidden_dim = h_dim
        self.output_dim = output_dim
        input_dim = np.shape(Examples_X)[1]
        np.random.seed(0)
        W1 = np.random.randn(input_dim, self.hidden_dim) / np.sqrt(input_dim)
        b1 = np.zeros((1, self.hidden_dim))
        W2 = np.random.randn(self.hidden_dim, self.output_dim) / np.sqrt(self.hidden_dim)
        b2 = np.zeros((1, self.output_dim))
        self.model = {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
        
    def calculate_loss(self):
        num_examples = np.shape(self.training_set_X)[0]
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        z1 = self.training_set_X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        corect_logprobs = -np.log(probs[range(num_examples), self.training_set_Y])
        data_loss = np.sum(corect_logprobs)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / num_examples * data_loss
    
    def train_model(self,num_passes,ifInit=False,ifSample = True,print_loss=False):
        W1 = self.model['W1']
        b1 = self.model['b1']
        W2 = self.model['W2']
        b2 = self.model['b2']
        for i in xrange(0, num_passes):
            if( i % 20000==0):
                #decay of learning rate
                self.epsilon  -= 0.1*self.epsilon
            if(ifSample):
                X,y = self.sample_data()
            else:
                X,y = self.training_set_X,self.training_set_Y
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
            delta3 = probs
            delta3[range(len(X)), y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)
    
            dW2 += self.reg_lambda * W2
            dW1 += self.reg_lambda * W1
    
            W1 += -self.epsilon * dW1
            b1 += -self.epsilon * db1
            W2 += -self.epsilon * dW2
            b2 += -self.epsilon * db2

            self.model = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}
            if(print_loss and i % 1000 == 0):
                print "Loss after iteration %i:%f" % (i, self.calculate_loss())
                
    def predict(self,x):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)  