from __future__ import division
import numpy as np
np.set_printoptions(suppress=True)

times = 10
N = 20 #number of agents
P = 100 #probability base 
food = np.ones(1000*times)
agent = np.zeros((N,P))
score = np.zeros(N)
score_test = np.zeros(N)
probability = np.random.randint(size=N,low=2,high=P)
time = np.random.randint(size=N,low=2,high=30)/30
score_one = (1/time)/np.max(1/time)

for num,(x,y) in enumerate(zip(agent,probability)):
    x[:y] = 1
    np.random.shuffle(x)

for x in food:
    a = np.arange(N)
    b = np.random.randint(size=N,low=0,high=P)
    
    base = agent[a,b]
    res = base * score_one
    index = np.where(res==np.max(res))
    score[index] += 1
    score_test[index] += 1
    res = np.logical_not(base) * score_one  
  #  index = np.where(res==np.max(res))
    score_test -= np.logical_not(base)
  #  score[index] -=1

print "Number",np.sum(score>0)

res = np.dstack((probability,time,score,score_test))
index =  np.argsort(res[:,:,2])
print res[0][index]

