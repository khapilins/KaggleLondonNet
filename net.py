import numpy as np
import json
import random

class net:
    sizes = []
    weights = []
    biases = []
    num_of_layers = 0
    eta = 0.2      
    lda = 2.5

    accuracies=[]
    validation_accuracies=[]
    train_accuracies=[]

    costs=[]
    validation_costs=[]
    train_costs=[]
    
    def __init__(self,sizes):
        self.sizes = sizes
        self.num_of_layers = len(sizes)
        self.weight_initializer()


    def weight_initializer(self):
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) / np.sqrt(x) 
                      for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y,1) for y in self.sizes]
        self.weights = [np.random.randn(y,x)
                      for x,y in zip(self.sizes[:-1],self.sizes[1:])]

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = fcns.sigmoid(np.dot(w,a) + b)
        return a

    def SGD(self, training_data,test_data,
            epochs,
            mini_batch_size=1,            
            validation_data=None):
        """training data is represented as tuple (x,y)"""
        
        n = len(training_data)  
                              
        for j in xrange(epochs):            
            if mini_batch_size!=1:
                random.shuffle(training_data)
                mini_batches = [training_data[k:k + mini_batch_size]
                              for k in xrange(0,n,mini_batch_size)]
            else: mini_batches=[training_data]
            for batch in mini_batches:
                self.update_batch(batch,len(training_data))

            print "Epoch completed %s " %j            
           
            self.costs.append(self.total_cost(test_data))
           
            if validation_data:
                self.validation_accuracies.append(self.accuracy(validation_data))
                print "Validation accuracy %s"%self.validation_accuracies[-1]
                self.validation_costs.append(self.total_cost(validation_data))

            if not validation_data:
                self.accuracies.append(self.accuracy(test_data))
                if self.accuracies[-1]>self.best_net_accuracy:
                    self.best_net_accuracy=self.accuracies[-1]
                    self.best_net=self.copy_net_to(self.best_net)
                    stop_counter=0
                else: stop_counter=stop_counter+1
            else:
                if self.validation_accuracies[-1]>self.best_net_accuracy:
                    self.best_net_accuracy=self.validation_accuracies[-1]
                    self.best_net=self.copy_net_to(self.best_net)
                    stop_counter=0                    
                else: stop_counter=stop_counter+1
            self.train_accuracies.append(self.accuracy(training_data))
            self.train_costs.append(self.total_cost(training_data))
            
            if self.train_accuracies[-1]>0.998:
                break
                    
        self.save()               
                     
    def cross_validate(self,training_data,test_data,epochs, k=3):
        l=len(training_data)
        to_train=[]        
        for i in xrange(k):            
            to_train.append((training_data[i*l/k:(i+1)*l/k],(training_data[:i*l/k]+(training_data[(i+1)*l/k:]))))            

        #to_train=[(training_data[i:i+k],training_data[i+k:]) for i in xrange(0,k,l)]
        for data in to_train:
            print '--------------------------------------'
            self.SGD(data[0],test_data,epochs,1,0,data[1])

    def update_batch(self,batch,n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in batch:
            delta_nabla_b,delta_nabla_w=self.backpropagate(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-self.eta*(self.lda/n))*w-(self.eta/len(batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(self.eta/len(batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def predict(self,data):
        return np.argmax(self.feedforward(data))

    def backpropagate(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        activation=x

        activations=[x]
        zs=[]
        for b,w in zip (self.biases, self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation=fcns.sigmoid(z)
            activations.append(activation)
        
        delta=fcns.delta(zs[-1],activations[-1],y)
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta,activations[-2].transpose())

        for l in xrange(2,self.num_of_layers):
            z=zs[-l]
            sp=fcns.sigmoid_prime(z)
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
            res.append(fcns.cross_entropy_cost(a,y))
        return np.sum(res)/float(len(test_data))

    def copy_net_to(self,copy_to_net):
        copy_to_net=net([])
        copy_to_net.sizes=self.sizes
        copy_to_net.biases=self.biases
        copy_to_net.weights=self.weights
        copy_to_net.num_of_layers=self.num_of_layers
        copy_to_net.best_net_accuracy=self.best_net_accuracy
        return copy_to_net

    def save(self):
        """Saving net to json"""
        f=open(str(self.best_net_accuracy)+'_best_net.json','wb')
        data={'sizes':self.sizes,
              'weights':[w.tolist() for w in self.weights],
              'biases':[b.tolist() for b in self.biases],
              'best_net_accuracy':self.best_net_accuracy}
        json.dump(data,f)
        f.close()

    def load(self,filename):
        """Loading net from json"""
        f=open(filename,'r')
        data=json.load(f)
        f.close()
        n=net([])
        n.sizes=data['sizes']
        n.weights=[np.array(w) for w in data['weights']]
        n.biases=[np.array(b) for b in data['biases']]
        n.best_net_accuracy=data['best_net_accuracy']
        return n

    @staticmethod
    def get_data_for_testing(file):
        pass


class fcns:

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        return fcns.sigmoid(x) * (1 - fcns.sigmoid(x))

    @staticmethod
    def cross_entropy_cost(a,y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z,a,y):
        return np.subtract(a, y)
