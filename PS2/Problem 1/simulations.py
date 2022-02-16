import numpy as np
import matplotlib.pyplot as plt

def IVP():
    x = np.arange(0,L,dL)
    return u0/(1+np.exp(x - x0))

if __name__ == '__main__':
    #given
    rho = 0.5
    q = 8
    L = 100
    u0 = 50    #CHANGE
    x0 = 20

    #user defined
    dL = 1
    T = 10
    dt = 0.1
    steps = int(T/dt)

    #initialize
    u = np.zeros((L,steps))
    u[:,0] = IVP()
    #no boundary flux
    u[0,:] = np.copy(u[0,0])
    u[L-1,:] = np.copy(u[L-1,0])

    #print(IVP())
    for t in range(steps-1):
        for i in range(1,L-1):
            u[i,t+1] = u[i,t] + dt*(rho*u[i,t]*(1 - u[i,t]/q) - u[i,t]/(1 + u[i,t]) + (u[i+1,t] + u[i-1,t] - 2*u[i,t])/dL**2)

    #plt.plot(u[:,0])
    #plt.plot(u[:,steps-1])
    plt.imshow(u, cmap='hot')
    plt.colorbar(orientation='horizontal')
    plt.show()
    

    
