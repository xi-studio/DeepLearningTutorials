import numpy as np

N = 20 # nubmer of neurons
Mes = 3 # number of messages
Mass = 40 # mass of each message's food
Tent = 5 # max number of each neuron's tentacle 

food = np.random.choice([-1,1],(Mes,Mass)) # A = 1,B = -1
food = np.ones((Mes,Mass))
food[:,:16] = -1
food_p = (food.reshape(Mes * Mass))[:Tent*N].reshape((N,Tent))

def eat():
    portion = np.zeros((N,Tent))
    portion = food_p 
    np.random.shuffle(portion)
    W = np.random.choice([0,1],(N,Tent))
    b = np.zeros(N)
    
    res = np.sum(W * portion,axis=1)+b
    res = (res>0)*1
    return np.sum(res)>N/2


if __name__ == '__main__':
    num = 0
    for x in range(100):
        num += eat() 
    print num
