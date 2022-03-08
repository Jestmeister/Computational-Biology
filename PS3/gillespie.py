from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    b_n = 0.1
    d_n = 0.2
    pop = 100
    n_runs = 50000
    dt = 0.01
    freq_b = []
    freq_d = []

    for i in range(n_runs):
        n = pop
        t = dt
        while n == pop:
            r = np.random.rand()
            if r < b_n*dt:
                n = n+1
                freq_b.append(t)
            t += dt
    
    for i in range(n_runs):
        n = pop
        t = dt
        while n == pop:
            r = np.random.rand()
            if r < d_n*dt:
                n = n+1
                freq_d.append(t)
            t += dt

    print('mean(b_n) = '+str(sum(freq_b)/len(freq_b)))
    print('mean(d_n) = '+str(sum(freq_d)/len(freq_d)))

    plt.figure()
    plt.hist(freq_d,bins=int(1/dt),density=True,label='P(t_d)')
    t_max = max(freq_d)
    x = np.linspace(0,t_max,n_runs)
    y = np.exp(-d_n*x)
    plt.plot(x,y,label='-d_n*t')
    plt.yscale('log')
    plt.legend()
    
    plt.figure()
    plt.hist(freq_b,bins=int(1/dt),density=True,label='P(t_b)')
    t_max = max(freq_b)
    x = np.linspace(0,t_max,n_runs)
    y = np.exp(-b_n*x)
    plt.plot(x,y,label='-b_n*t')
    plt.yscale('log')
    plt.legend()
    plt.show()