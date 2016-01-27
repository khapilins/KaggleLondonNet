import numpy as np

class sigmoid:
    @staticmethod
    def f(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def f_prime(x):
        return sigmoid.f(x) * (1 - sigmoid.f(x))

class tanh:
    @staticmethod
    def f(x):        
        return np.tanh(x)        

    @staticmethod
    def f_prime(x):
        return 1-np.square(tanh.f(x))

class ReLU:
    @staticmethod
    def f(x):                   
        b=np.maximum(0.,x)
        return b

    @staticmethod
    def f_prime(x):        
        xx=x.copy()
        xx/=np.abs(x)
        return np.maximum(0.3,xx)