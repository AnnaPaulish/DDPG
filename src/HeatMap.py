import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib as mpl
import torch
# Network input: shape: (batch_size, 4). sample: [x,y,theta' (= velocity),action]
#state = (next state, reward, term, trunc, info)

def Multiple_HeatMap(Neural_Network, Action = 0, Velocity = 0):
    
    nr_of_states = 500
    fig = plt.figure(figsize=(15,5))
    vmin = 0
    vmax = 0
    for i in np.arange(len(Action)):
        ########
        action = np.zeros((nr_of_states,1)) + Action[i]
        velocity = np.zeros((nr_of_states,1)) + Velocity[i]
        q_value = []
        theta = np.reshape(np.linspace(0,2*np.pi,nr_of_states),(nr_of_states,1))
        x = np.cos(theta)
        y = np.sin(theta)

        state = np.concatenate((x,y,velocity,action),1)

        with torch.no_grad():
            state = torch.tensor(state, dtype = torch.float32)
            new_q_value = Neural_Network(state)
            

        rad = np.linspace(0, 1, nr_of_states)
        azm = np.linspace(0, 2 * np.pi, nr_of_states)
        r, th = np.meshgrid(rad, azm)

        z = np.repeat(new_q_value, [nr_of_states], axis=1)
        #######3

        cmap = mpl.cm.plasma

        ax = plt.subplot(1, len(Action), i+1, projection = 'polar')
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        ax.set_title("Velocity = {}, Action = {}".format(Velocity[i], Action[i]))
        ax.pcolormesh(th, r, z, cmap=cmap)
        ax.plot(th, r, color='k', ls='none') 
        ax.grid()
        vmin = min(vmin, torch.min(z))
        vmax = max(vmax, torch.max(z))


    fig.suptitle("q value as a function of the pendulum angle", x = 0.5, y = 0.85)


    cax = plt.axes([0.95, 0.25, 0.05, 0.55])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax = cax) 
    cbar.set_label('q value')
    plt.show()





def Single_HeatMap(Neural_Network, Action = 0, Velocity = 0):

    nr_of_states = 500

    action = np.zeros((nr_of_states,1)) + Action
    velocity = np.zeros((nr_of_states,1)) + Velocity
    q_value = []
    theta = np.reshape(np.linspace(0,2*np.pi,nr_of_states),(nr_of_states,1))
    x = np.cos(theta)
    y = np.sin(theta)

    state = np.concatenate((x,y,velocity,action),1)

    with torch.no_grad():
        state = torch.tensor(state, dtype = torch.float32)
        new_q_value = Neural_Network(state)
        

    rad = np.linspace(0, 1, nr_of_states)
    azm = np.linspace(0, 2 * np.pi, nr_of_states)
    r, th = np.meshgrid(rad, azm)

    z = np.repeat(new_q_value, [nr_of_states], axis=1)


    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=torch.min(z), vmax=torch.max(z))

    fig = plt.figure()

    ax = plt.subplot(projection='polar')
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.pcolormesh(th, r, z, cmap=cmap)
    ax.plot(th, r, color='k', ls='none') 
    ax.grid()
    ax.set_yticklabels([])
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.title("q value as a function of the pendulum angle with velocity = {} and action = {}".format(Velocity, Action), x = -3, y = 1.05)

    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax = cax) 
    cbar.set_label('q value')
    plt.show()
    
