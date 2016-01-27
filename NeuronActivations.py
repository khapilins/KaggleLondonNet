from Activations import *

class NeuronActivations:
    """
    Class contains dictionary of activation functions and their primes
    were key is a string and value is class with 2 static methods f(x) and f_prime(x)
    """

    activations={
        'sigmoid':sigmoid,
        'tanh':tanh,
        'ReLU':ReLU
        }
