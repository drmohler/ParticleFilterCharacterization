#RSR Test Script
from math import *
import random

def RS_resample(N,weights, particles):
    p_new = []
    index = [0]*N #Initialize index array

    U = random.random()/N #Generate a random number between 0 and 1/N
    i  = 0
    j = -1

    while j < N-1:
        j += 1
        Ns = floor(N*(weights[j]-U))+1
        counter = 1;
        while counter <= Ns:
            index[i] = j
            i += 1
            counter += 1
        U = U + Ns/N - weights[j]

    for i in range(len(index)):
        p_new.append(particles[index[i]])

    particles = p_new
    shrtWeights = ["%.3f" % elem for elem in weights]
    print(index)
    return particles


particles = [[2,3],[6,2],[7,7],[1,0],[9,5]]
w = [0.21,0.082,0.42,0.11,0.178]
N = 5
print("Sum: ", sum(w))
print("Original Particles: ", particles)
particles = RS_resample(N,w,particles)

print("Resampled Particles: ", particles)
