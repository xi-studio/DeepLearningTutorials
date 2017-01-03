import numpy as np

N = 1 # nubmer of neurons
Mes = 8 # number of messages
Mass = 10 # mass of each message's food
Tent = Mes*Mass/N # max number of each neuron's tentacle 


pic = np.zeros(Mass)

def io2food(xi):
    res = np.unpackbits(np.array([xi],dtype=np.uint8))
    food = np.outer(res,pic) # A = 1, B= -1
    return food.reshape(Mes * Mass)

def getTent():
    tent = np.zeros((N,Mes,N,Mass/N))
    for x in range(N):
        tent[x][:,x,:] = 1

    return tent.reshape((N,Mes*Mass))

def eat(food_p):
    portion = np.zeros((N,Tent))
    portion = food_p[:Tent*N].reshape((N,Tent))
    W = np.random.choice([0,1],(N,Tent))
    b = np.zeros(N)
    
    res = np.sum(W * portion,axis=1)+b
    res = (res>0)*1
    return np.sum(res)>N/2


def run(food_p,iters=1000):
    num = 0
    for x in range(iters):
        num +=eat(food_p)
    return num
    

if __name__ == '__main__':
    pic = np.random.choice([-1,1],Mass)
    for x in range (10):
        food = io2food(x)
        print food
        tent = getTent()
        print np.sum(food * tent ,axis=1)>0
