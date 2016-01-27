import numpy as np
import json
import random
import NeuronActivations
import CostFunctions

class net:
    sizes = []
    weights = []
    biases = []
    num_of_layers = 0
    eta = 0.1
    lda = 0.

    accuracies=[]
    validation_accuracies=[]
    train_accuracies=[]

    costs=[]
    validation_costs=[]
    train_costs=[]
    
    def __init__(self,sizes,activation, cost):
        """
        activation and cost parameters are used to get 
        needed activation and cost function classe from NeuronActivation.py and CostFunctions.py
        ~~~~~~~~~~~~~~~~~~~~~~
        for example:
        n=net([40,20,5],'sigmoid','cross_entropy_loss')
        """
        self.sizes = sizes
        self.num_of_layers = len(sizes)
        self.weight_initializer()
        self.activation=NeuronActivations.NeuronActivations.activations[activation]
        self.cost=CostFunctions.CostFunctions.costs[cost]


    def weight_initializer(self):
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) * np.sqrt(2./x) 
                      for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y,1) for y in self.sizes]
        self.weights = [np.random.randn(y,x)
                      for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = self.activation.f(np.dot(w,a) + b)
        return a

    def SGD(self, training_data,test_data,
            epochs,
            mini_batch_size=1,            
            validation_data=None,
            max_accuracy_on_train_data=0.998,
            use_momentum=True,
            mu=0.9,
            filename_to_save=''):
        """
        training data is represented as tuple (x,y)
        with mini_batch_size of 1 batches are not shuffled
        so SGD works like regular GD
        ~~~~~~~~~~~~~~~~~~
        when accuracy on train data equals or above max_accuracy_on_train_data
        learning stops to prevent overfitting
        """
        
        n = len(training_data)  
                              
        for j in xrange(epochs):            
            if mini_batch_size!=1:
                random.shuffle(training_data)
                mini_batches = [training_data[k:k + mini_batch_size]
                              for k in xrange(0,n,mini_batch_size)]
            else: mini_batches=[training_data]
            for batch in mini_batches:
                self.update_batch(batch,len(training_data),use_momentum,mu)

            print "Epoch completed %s " %j            
           
            self.costs.append(self.total_cost(test_data))
           
            if validation_data:
                self.validation_accuracies.append(self.accuracy(validation_data))
                print "Validation accuracy %s"%self.validation_accuracies[-1]
                self.validation_costs.append(self.total_cost(validation_data))
            
            self.train_accuracies.append(self.accuracy(training_data))
            self.train_costs.append(self.total_cost(training_data))
            
            if self.train_accuracies[-1]>max_accuracy_on_train_data:
                break
                    
        #self.save(filename_to_save)               
                         

    def update_batch(self,batch,n, use_momentum=True,mu=0.9):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #v is velocity for momentum
        if use_momentum:
            v=[np.zeros(w.shape) for w in self.weights]        
        for x,y in batch:
            delta_nabla_b,delta_nabla_w=self.backpropagate(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            if use_momentum:
                v=[mu*vv-self.eta/len(batch)*nw for vv,nw in zip(v,nabla_w)]

        if  use_momentum:
            self.weights=[w+nw-self.eta*(self.lda/n)
                          for w, nw in zip(self.weights, v)]
        else:
            self.weights = [(1-self.eta*(self.lda/n))*w-(self.eta/len(batch))*nw
                            for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(self.eta/len(batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
  
    def backpropagate(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation=x

        activations=[x]
        zs=[]
        for b,w in zip (self.biases, self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=self.activation.f(z)
            activations.append(activation)
        
        delta=self.cost.loss_derivative(zs[-1],activations[-1],y)
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())

        for l in xrange(2,self.num_of_layers):
            z=zs[-l]
            sp=self.activation.f_prime(z)
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sp

            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())

        return nabla_b,nabla_w    

    def accuracy(self,test_data):        
        results = [(np.argmax(self.feedforward(x)),np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)/float(len(test_data))

    def total_cost(self,test_data):
        As=[(self.feedforward(x),y) for x,y in test_data ]
        res=[]
        for a,y in As:
            res.append(self.cost.cost(a,y))
        return np.sum(res)/float(len(test_data))

    def cross_validate(self,training_data,test_data,epochs, k=3,mini_batch_size=10):
        l=len(training_data)
        to_train=[]        
        for i in xrange(k):            
            to_train.append((training_data[i*l/k:(i+1)*l/k],(training_data[:i*l/k]+(training_data[(i+1)*l/k:]))))            
        
        for data in to_train:
            print '--------------------------------------'
            self.SGD(data[0],test_data,epochs,mini_batch_size,data[1])

    def predict(self,data):
        return np.argmax(self.feedforward(data))
