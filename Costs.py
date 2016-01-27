import numpy as np

class square_loss:
    @staticmethod
    def cost(a,y): 
        return np.sum(np.square(y-a))        

    @staticmethod
    def loss_derivative(z,a,y):
        return (a-y)

class cross_entropy_loss:
    @staticmethod
    def cost(a,y):                 
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def loss_derivative(z,a,y):
        return np.subtract(a, y)

class hinge_loss:
    @staticmethod
    def cost(a,y):         
        return np.sum(np.maximum(0.,1.-y*a))        

    @staticmethod
    def loss_derivative(z,a,y):        
        return np.maximum(0.3, 1.-y)

class scaled_cross_entropy_loss:
    """
    scaled case of cross entropy for tanh(x)
    """
    @staticmethod
    def cost(a,y):                 
        return np.sum(np.nan_to_num(-(y+1.)/2. * np.log((a+1.)/2.+0.01) - (1 - (y+1.)/2.) * np.log(1.01 - (a+1.)/2.)))

    @staticmethod
    def loss_derivative(z,a,y):        
        sub=a-y
        delim=a*a-1.        
        delim=np.maximum(0.01,delim)
        return sub/delim