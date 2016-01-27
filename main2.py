from net2 import *
from matplotlib import pyplot as plt
import random
import os

f1=open('train.csv')
f2=open('trainLabels.csv')
x=f1.readlines()
y=f2.readlines()
xs=[]
ys=[]
for line in x:
    xs.append(np.array([float(param) for param in line.split(',')]).reshape((40,1)))

mean=np.mean(np.mean(xs[:700],axis=0))
#std=np.std(xs,axis=0)
max=np.max(np.max(xs[:700]))
min=np.min(np.min(xs[:700]))

if abs(min)>abs(max):
    max=min

xs-=mean
xs/=abs(max)-mean

max=np.max(xs)
min=np.min(xs)
for line in y:
    if float(line)==1:
        ys.append(np.array([0,1]).reshape((2,1)))
    else:
        ys.append(np.array([1,0]).reshape((2,1)))

train_data=[(tx,ty) for tx,ty in zip(xs,ys)]

f3=open('test.csv','r')
ts=[]
for line in f3:
    ts.append(np.array([float(param) for param in line.split(',')]).reshape((40,1)))

random.shuffle(train_data)

np.seterr(all='raise')

train=train_data[:700]
valid=train_data[701:900]
test=train_data[901:]

n=net([40,35,15,2],'sigmoid','cross_entropy_loss')
n.eta=0.01

n.SGD(train,test,500,10,validation_data=valid,use_momentum=True)

t,a=plt.subplots(2)
a[0].plot(n.accuracies)
a[0].plot(n.validation_accuracies)
a[0].plot(n.train_accuracies)

a[1].plot(n.costs)
a[1].plot(n.validation_costs)
a[1].plot(n.train_costs)

print "Expecting accuracy %s"%n.accuracy(test)

#print 'creating test labels csv'
#f=open('testLabels.csv','w')
#f.write("Id,Solution\n")
#for i in xrange(len(ts)):
#    f.write(str(i+1)+','+str(n.predict(ts[i]))+'\n')    
#f.close()

plt.show()

