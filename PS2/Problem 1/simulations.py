import numpy as np
import matplotlib.pyplot as plt

def IVP():
    x = np.arange(0,L,dL)
    return u0/(1+np.exp(x - x0))

def calcFP(u):
    J = [[0, 1],[-rho + 2*rho/q*u + 1/(1+u)**2,-c]]
    print('Trace: ' + (J[0,0] + J[1,1]))

if __name__ == '__main__':
    #given
    rho = 0.5
    q = 8
    u_star_1 = (q-1)/2 + np.sqrt((q-1)**2/4 + q*(1-1/rho))  #Largest steady-state
    u_star_2 = (q-1)/2 - np.sqrt((q-1)**2/4 + q*(1-1/rho))  #2nd Largest steady-state
    L = 100
    u0 = u_star_1
    x0 = 20

    #user defined
    dL = 1
    T = 200
    dt = 0.01
    steps = int(T/dt)
    L_fix = 50
    t_fix = int(50/dt)
    t1 = 0
    t2 = 0
    u1 = 0
    u2 = 0
    uThresh = 0.1

    #initialize
    u = np.zeros((L,steps))
    du_dEps = np.zeros((L,1))
    u[:,0] = IVP()
    #no boundary flux
    u[0,:] = np.copy(u[0,0])
    u[L-1,:] = np.copy(u[L-1,0])

    #print(IVP())
    for t in range(steps-1):
        for i in range(1,L-1):
            u[i,t+1] = u[i,t] + dt*(rho*u[i,t]*(1 - u[i,t]/q) - u[i,t]/(1 + u[i,t]) + (u[i+1,t] + u[i-1,t] - 2*u[i,t])/dL**2)
        if u[L_fix,t] > uThresh and t1 == 0:
            t1 = np.copy(t)
        if u[L_fix + 1,t] > uThresh and t2 == 0:
            t2 = np.copy(t)

    for i in range(1,L-1):
        du_dEps[i] = (u[i+1,t_fix] - u[i-1,t_fix])/ (2*dL)

    #Find c
    print(t1)
    print(t2)
    c = 1/(dt*(t2 - t1))
    print(c)
    
    #Calc matrix
    calcFP(0)
    calcFP(u0)

    plt.figure()
    plt.subplot(121)
    plt.plot(u[:,t_fix])
    plt.xlabel('\u03BE')
    plt.ylabel('u')
    plt.subplot(122)
    plt.plot(u[:,t_fix],du_dEps)
    plt.xlabel('u')
    plt.ylabel('du/d\u03BE')
    
    #plt.figure()
    #plt.plot(u[:,steps-1])
    
    #plt.figure()
    #plt.imshow(u, cmap='hot')
    #plt.colorbar(orientation='horizontal')
    plt.show()
    
    

    
