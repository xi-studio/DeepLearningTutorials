import numpy as np

N = 20 # nubmer of neurons
Mes = 3 # number of messages
Mass = 40 # mass of each message's food
Tent = 6 # max number of each neuron's tentacle 


def io2food(xi):
    food = np.outer(xi, np.random.choice([-1,1],Mass)) # A = 1, B= -1
    return food

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
    xi = np.array([0,0,1])
    food = io2food(xi)
    food_p = (food.reshape(Mes * Mass))
    for x in range(100):
        np.random.shuffle(food_p)
        print run(food_p)
