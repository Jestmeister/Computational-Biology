import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 50 #multiple N:s
    gamma = 0.7
    omega = cauchy.rvs(loc=0,scale=gamma,size=N)
    Kc = 2*gamma
    #K = Kc/3 
    #K = 5*Kc
    K = Kc + 0.1
    r_ref = np.sqrt((K-Kc)/Kc)
    print(r_ref)

    dt = 0.01
    T = 10*N/K
    t_steps = int(T/dt)

    theta = np.zeros((N,t_steps))
    theta[:,0] = np.pi/2*np.random.uniform(-1,1,N)
    r = np.zeros((t_steps,1))

    #Euler forward
    for t in range(t_steps-1):
        for i in range(N):
            theta[i,t+1] = theta[i,t] + dt * (
            omega[i] + K/N*np.sum(np.sin(theta[:,t] - theta[i,t]))) 

    plt.figure()
    plt.plot(theta.T)
    plt.xlabel('t')
    plt.ylabel('theta')

   #Simulate r
    for t in range(t_steps):
        big_sum = 0
        for i in range(N):
            for j in range(i+1,N):
                big_sum += np.cos(theta[i,t] - theta[j,t])
        temp = (1 + 2*big_sum)/N
        if temp < 0:
            r[t] = 0
        else:
            r[t] = np.sqrt(temp)

    plt.figure()
    plt.plot(r)
    plt.xlabel('t')
    plt.ylabel('r')
    plt.show()