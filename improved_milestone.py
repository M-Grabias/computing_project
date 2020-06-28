from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.gridspec as gridspec
import matplotlib.colors
from matplotlib.colors import BoundaryNorm
from matplotlib import animation



def lattice_creator(N, spins):
    '''Function returning a square lattice of size N filled with randomly oriented spins as a numpy array'''

    lattice = np.zeros( (N,N) ) #create an empty square lattice of size N
    for y in range(N):
        for x in range(N):
            lattice[y,x] = int(random.choice(spins)) #fill the whole lattice with random spins

    return lattice


def magnetization(N, lattice):
    '''Function returning the average magnetization per spin of the input square lattice as a numpy array of size N'''

    return ( 1 / (N**2) ) * np.sum(lattice)


def phase_transition(N, J, B, lattice):
    '''Funtion simulating the phase transition of a 2D Ising model using the Metropolis algorithm'''

    for n in range(N**2):

        x, y = random.choice( range(N) ), random.choice( range(N) )

        neighbour_spins = np.zeros(4)     #initiate an array to store the values of 4 nearest neighbouring spins (up/down, right/left)

        if x-1 >= 0:             #check whether indices of neighbours are within lattice

            neighbour_spins[0] = lattice[y,x-1]        #if so include them in neighbour array

        if x+1 <= N - 1:
                
            neighbour_spins[1] = lattice[y,x+1]

        if y-1 >= 0:

            neighbour_spins[2] = lattice[y-1,x]

        if y+1 <= N - 1:

            neighbour_spins[3] = lattice[y+1,x]

        sum_neighbouring_spins = np.sum(neighbour_spins) #calculate the sum of all 4 neighbouring spin values

        delta_e = 2 * (J * sum_neighbouring_spins + B) * (-1) * lattice[y,x] #calculate the energy change for this spin flip

                
                #Metropolis Algorithm start here:
        if np.exp(delta_e) > 1:

            lattice[y,x] = - lattice[y,x]     #retain this new spin

        elif np.exp(delta_e) > random.random():

            lattice[y,x] = - lattice[y,x]     #also retain this new spin

        else:

            pass #keep the original spin and do not flip, ie. do nothing

    return lattice

def animation_update(frame, N, J, B, lattice, image, axis):

    if frame == 0:

        image.set_data(lattice)
        plt.xlabel('MC steps = 0', fontsize='20')

    else:

        update = phase_transition(N, J, B, lattice)
        image.set_data(update)
        axis.set_xlabel('MC steps = {}'.format(frame + 1), fontsize='20')

    



if __name__ == "__main__":

    J = 0.5
    B = -0.1
    N = 100
    Spins = [-1, 1]

    Lattice = lattice_creator(N, Spins) # generate a random lattice 


    #### create a plot ####

    fig = plt.figure()


    cmap=plt.cm.summer #get nice colours

    spec = gridspec.GridSpec(ncols=2, nrows=1)
    
    
    ax1 = fig.add_subplot(spec[0, 0]) # add 'still' snapshot of the initial randomly generated lattice
    im1 = ax1.imshow(Lattice, cmap=cmap, interpolation='nearest')
    ax1.yaxis.set_major_locator(plt.NullLocator())
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    plt.xlabel('MC steps = 0', fontsize='20')
    

    ax2 = fig.add_subplot(spec[0, -1])   # add animated plot showing how the lattice changes
    im2 = ax2.imshow(Lattice, cmap=cmap,  interpolation='nearest')

    interval = 200 #delay between frames in ms
    anim = animation.FuncAnimation(fig, animation_update, interval=interval, fargs = [N, J, B, Lattice, im2, ax2])

    ax2.yaxis.set_major_locator(plt.NullLocator())
    ax2.xaxis.set_major_formatter(plt.NullFormatter())

    
    plt.show()


