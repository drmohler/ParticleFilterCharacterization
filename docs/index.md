### Introduction to Particle Filters:

"Particle" methods, initially introduced in 1993,  are a class of numerical
methods which are popular in application to the estimation of problems that are
either highly non-linear or non-Gaussian in nature. Through a nonparametric implementation of the Bayes filter, the particle filter attempts to approximate the posterior distribution of the state of the system using a finite number of randomly drawn state samples. These samples drawn from the posterior, known as "particles",  each represent a hypothesis as to the true state of the system. Each particle is weighted proportionally to the probability that it represents the true state according to some proposal distribution. The combination of the particles and their weights form the estimated probability distribution. An example particle cloud at initialization is shown in the figure below. The particles carrying an "effective" probability weight are highlighted in green, with the true state in blue, and the estimated state marked in black.

  ![Particle Cloud](graphics0_1.png)

## Particle Filtering
 This project explores two methods for the estimation of the state with particle methods developed in python. The first being the traditional style of particle filtering which incorporates importance resampling methods, and the second, applying log homotopy particle flow filtering methods as proposed by Duam and Huang ("Nonlinear filters with log-homotopy", 2007).

### Generic Algorithm for Traditional Resampling Particle Filters

1. **Propagate particles**
  * Move the particles according to the control input and model the uncertainty in the system

  ```python
  def move(self,turn,forward):
      """
      turn: variable describing the change in heading (radians)
      forward: robots present velocity
      """
      if forward < 0:
          raise ValueError('Robot can only move forward')

      #turn, and add randomness to the command
      hdg = self.hdg + float(turn) + random.gauss(0.0,self.turn_noise)
      hdg %= 2*np.pi
      dist = float(forward) + random.gauss(0.0,self.forward_noise)

      #Define x and y motion based upon new bearing relative to the x axis
      x = self.x + (cos(hdg)*dist)
      y = self.y + (sin(hdg)*dist)
      x %= self.world_size #cyclic truncate
      y %= self.world_size

      #set particles
      res = robot()
      res.set_params(self.N,self.world_size,self.landmarks)
      res.set(x,y,hdg) # changes particle's position to the new location
      res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
      return res
  ```
2. **Update**
  * asdf


 As can be seen in the figure, the present set of particles offer a poor representation of the actual state of the system.


![GIF](ParticleFilterTracking.gif)
