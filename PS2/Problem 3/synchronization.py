from cProfile import label
import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 100 #multiple N:s
    gamma = 0.4
    omega = cauchy.rvs(loc=0,scale=gamma,size=N)
    Kc = 2*gamma
    K_list = np.array([Kc/3, Kc + 0.01, 1.5*Kc])
    r_ref = np.sqrt((K_list-Kc)/Kc)
    print(r_ref)

    dt = 0.01
    T = int(3*N/K_list[0])
    #T = int(N/K_list[0])
    t_steps = int(T/dt)
    r = np.zeros((T,3))

    count = 0
    for K in K_list:
        print(K)
        theta = np.zeros((N,t_steps))
        theta[:,0] = np.pi/2*np.random.uniform(-1,1,N)
        
        #Euler forward
        for t in range(t_steps-1):
            for i in range(N):
                theta[i,t+1] = theta[i,t] + dt * (
                omega[i] + K/N*np.sum(np.sin(theta[:,t] - theta[i,t]))) 
                #Stay in circle
                if theta[i,t+1] < -np.pi:
                    theta[i,t+1] += 2*np.pi
                elif theta[i,t+1] > np.pi:
                    theta[i,t+1] -= 2*np.pi

        #plt.figure()
        #plt.plot(theta.T)
        #plt.xlabel('t')
        #plt.ylabel('theta')

        #Simulate r
        #for t in range(t_steps):
        for k in range(T):
            t = int(k/dt)
            big_sum = 0
            for i in range(N):
                for j in range(i+1,N):
                    big_sum += np.cos(theta[i,t] - theta[j,t])
            temp = N + 2*big_sum
            if temp < 0:
                r[k,count] = 0
            else:
                r[k,count] = np.sqrt(temp)/N
        count += 1
        #r_mean = np.mean(r)

    #change colors, 
    plt.figure()
    colors = ['tab:blue','tab:green','tab:orange']
    label_list = [r'K_1<K_c',r'K_2>K_c',r'K_3>K_c']
    label_list2 = [r'ogga','Mean-field '+r'K_2>K_c','Mean-field'+r'K_3>K_c']
    colors2 = ['','k','r']
    for i in range(3):
        plt.plot(r[:,i], label=label_list[i],color=colors[i])
        if i != 0:
            plt.plot([0,T],[r_ref[i],r_ref[i]],'--',label=label_list2[i], color=colors2[i])
    #plt.plot([0,T],[r_mean,r_mean], label='r sim mean')
    plt.xlabel('t')
    plt.ylabel('r')
    plt.legend(loc='lower left')
    plt.show()