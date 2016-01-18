from sklearn import svm
import numpy as np

f1=open('train.csv')
f2=open('trainLabels.csv')
x=f1.readlines()
y=f2.readlines()
xs=[]
ys=[]
for line in x:
    xs.append(([float(param) for param in line.split(',')]))

for line in y:
    ys.append(float(line))

clf=svm.SVC()
clf.fit(xs,ys)

res=[]
for x,y in zip(xs[:900],ys[:900]):
    res.append(round(clf.predict(x))==y)

print sum(res)/float(len(res))

res1=[]
for x,y in zip(xs[:900],ys[:900]):
    res1.append(round(clf.predict(x))==y)

print sum(res1)/float(len(res1))


f3=open('test.csv','r')
ts=[]
for line in f3:
    ts.append(([float(param) for param in line.split(',')]))

print 'creating test labels csv'
f=open('svmTestLabels.csv','w')
f.write("Id,Solution\n")
for i in xrange(len(ts)):
    f.write(str(i+1)+','+str(int(clf.predict(ts[i])[0]))+'\n')    
f.close()

