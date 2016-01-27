from Costs import *

class CostFunctions:
    """
    Class contains dictionary of cost functions and their primes
    were key is a string and value is class with 2 static methods cost(a,y) and cost_prime(z,a,y)
    where a - activation
    y - train labels
    z - weighted product w*x+b
    """

    costs={
        'square_loss':square_loss,
        'cross_entropy_loss':cross_entropy_loss,
        'hinge_loss':hinge_loss,
        'scaled_cross_entropy':scaled_cross_entropy_loss
        }
