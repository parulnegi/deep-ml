import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.alpha= smoothing
        self.feature_prob=[]
        self.prior=None
        self.classes=None


    def forward(self, X, y):
        self.classes= np.unique(y, return_counts=True)
        c0=np.sum(y==0)/len(y)
        c1= np.sum(y==1)/len(y)
        count_y0= X[y==0].sum(axis=0)
        prob_y0=(count_y0 + self.alpha)/(np.sum(y==0) + 2 * self.alpha)
        count_y1= X[y==1].sum(axis=0)
        prob_y1=(count_y1 + self.alpha)/(np.sum(y==1)+ 2 * self.alpha)

        self.feature_prob=np.vstack([prob_y0, prob_y1])
        self.prior=np.log([c0,c1])
        

    def predict(self, X):

        prob_x= X @ np.log(self.feature_prob.T) + (1-X) @ np.log(1-self.feature_prob.T)
        final_prob = prob_x + self.prior
        return np.argmax(final_prob, axis=1)












