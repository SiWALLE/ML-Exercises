from sklearn import datasets
import numpy as np
from NN_Classifier import NN_Classifier
from LR_Classifier import LR_Classifier


np.random.seed(0)
All_Exam_X, All_Exam_Y = datasets.make_moons(500, noise=0.2)

lr = LR_Classifier(All_Exam_X,All_Exam_Y)
lr.train_model()
lr.plot_decision_boundary()

nn = NN_Classifier(All_Exam_X,All_Exam_Y,h_dim=3,output_dim=2)
nn.train_model(20000,print_loss=True)
nn.plot_decision_boundary()

print 'hold on'
