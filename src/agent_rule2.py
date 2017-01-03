from __future__ import division
import numpy as np
np.set_printoptions(suppress=True)

N = 20 #number of agents
P = 100 #probability base 
deadline = 50
food = np.ones(100000)
agent = np.zeros((N,P))
score = np.zeros(N)
probability = np.random.randint(size=N,low=2,high=P)
time = np.random.randint(size=N,low=2,high=30)/30
score_one = (1/time)/np.max(1/time)

for num,(x,y) in enumerate(zip(agent,probability)):
    x[:y] = 1
    np.random.shuffle(x)

flag = np.zeros(N)
last = np.zeros(N)
iters = 0
for x in food:
    a = np.arange(N)
    b = np.random.randint(size=N,low=0,high=P)
    
    base = agent[a,b]
    res = base * score_one
    index = np.where(res==np.max(res))
    score[index] += 1
   
    flag += np.logical_not(base)
    index = np.where(flag<deadline)
    last[index] = score[index]
    index = np.where(flag==deadline)
    score_one[index] = 0
    score[index] = -1
    if np.sum(flag==deadline)>0:
        print probability[index]
    if np.sum(score_one>0)<1:
        break
     
    iters += 1 
    
    
print "Iters",iters
print "Number",np.sum(last>0)

res = np.dstack((probability,time,last))
index =  np.argsort(res[:,:,2])
print res[0][index]

