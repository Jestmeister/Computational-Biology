import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

def IVP():
    x = np.arange(0,L,dL)
    return u0/(1+np.exp(x - x0))

def IVP2():
    x = np.arange(0,L,dL)
    return u0*np.exp(-(x - x0)**2)

def calcFP(u):
    J = np.array([[0, 1],[-rho + 2*rho*u/q + 1/(1+u)**2,-c]])
    tr = J[0,0] + J[1,1]
    det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
    limit = tr**2 - 4*det
    print('Trace: ' + str(tr))
    print('Determinant: ' + str(det))
    print('Limit: ' + str(limit))

if __name__ == '__main__':
    #given
    rho = 0.5
    q = 8
    u_star_1 = (q-1)/2 + np.sqrt((q-1)**2/4 + q*(1-1/rho))  #Largest steady-state
    u_star_2 = (q-1)/2 - np.sqrt((q-1)**2/4 + q*(1-1/rho))  #2nd Largest steady-state
    L = 100
    #u0 = u_star_1
    #x0 = 20
    #u0 = u_star_2
    x0 = 50
    u0 = 1.1*u_star_2
    #u0 = 3*u_star_1

    #user defined
    dL = 1
    T = 120
    dt = 0.01
    steps = int(T/dt)
    #L_fix = 30
    #t_fix = int(50/dt)
    L_fix = 60
    t_fix = int(90/dt)
    t1 = 0
    t2 = 0
    u1 = 0
    u2 = 0
    uThresh = 0.1

    #initialize
    u = np.zeros((L,steps))
    du_dEps = np.zeros((L,1))
    u[:,0] = IVP()
    #u[:,0] = IVP2()

    for t in range(steps-1):
        for i in range(1,L-1):
            u[i,t+1] = u[i,t] + dt*(rho*u[i,t]*(1 - u[i,t]/q) - u[i,t]/(1 + u[i,t]) + (u[i+1,t] + u[i-1,t] - 2*u[i,t])/dL**2)
        #no boundary flux
        u[L-1,t+1] = u[L-1,t] + dt*(rho*u[L-1,t]*(1 - u[L-1,t]/q) - u[L-1,t]/(1 + u[L-1,t]) + (u[L-1-1,t] - u[L-1,t])/dL**2)
        u[0,t+1] = u[0,t] + dt*(rho*u[0,t]*(1 - u[0,t]/q) - u[0,t]/(1 + u[0,t]) + (u[0+1,t] - u[0,t])/dL**2)
        if u[L_fix,t] > uThresh and t1 == 0:    #rev sign
            t1 = np.copy(t)
        if u[L_fix + 1,t] > uThresh and t2 == 0:  #rev sign
            t2 = np.copy(t)

    for i in range(1,L-1):
        du_dEps[i] = (u[i+1,t_fix] - u[i-1,t_fix])/ (2*dL)

    #Find c
    print(t1)
    print(t2)
    if t1 == 0 and t2 == 0:
        c = 0
    else:
        c = 1/(dt*(t2 - t1))
    print(c)
    
    #Calc matrix
    calcFP(0)
    calcFP(u_star_1)

    plt.figure()
    plt.subplot(121)
    plt.plot(u[:,t_fix])
    plt.xlabel('\u03BE')
    plt.ylabel('u')
    plt.subplot(122)
    plt.plot(u[:,t_fix],du_dEps)
    plt.xlabel('u')
    plt.ylabel('du/d\u03BE')
    
    plt.figure()
    plt.plot(u[:,0])
    plt.xlabel('\u03BE')
    plt.ylabel('u')
    plt.figure()
    plt.plot(u[:,steps-1])
    plt.xlabel('\u03BE')
    plt.ylabel('u')
    
    plt.show()

    filenames = []
    for t in range(T):
        i = int(t/dt)
        # plot the line chart
        plt.plot(u[:,i])
        
        # create file name and append it to a list
        filename = f'{t}.png'
        filenames.append(filename)
        
        # save frame
        plt.savefig(filename)
        plt.close()
    # build gif
    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)