# ATRC 2017: Particle Filter Classification

This project was developed in support of the AFRL: Automatic Target Recognition Center summer internship, Summer 2017.

### Introduction to Particle Filters:

"Particle" methods, initially introduced in 1993,  are a class of numerical methods which are popular in application to the estimation of problems that are either highly non-linear or non-gaussian in nature.

![testing](PFTrack.png)

## Particle Filtering
#### Traditional Resampling Methods

```python
for tr in range(trials):
    p2=[]
    for i in range(n):
        #move the particles
        p2.append(p_m[m][tr][i].move(state[3],state[2]))
    p_m[m][tr] = p2

    w = []
    #generate particle weights based on measurement z
    for i in range(n):
        w.append(p_m[m][tr][i].measurement_prob(z))

    w_norm = []
    for i in range(n):
        # normalize the importance weights
        w_norm.append((w[i])/np.sum(w))

    #calculate the effective sample size
    neff = int(Robot.neff(w_norm))

    #if the effective sample size falls below 50% resample
    if neff < n/2:
        resample_count[tr] +=1
        #select method m to resample the particle set
        p_m[m][tr] = Robot.resample(n,w_norm,p_m[m][tr],methods[m])
        for i in range(n):
              w_norm[i] = 1/n

    #mean and covariance state estimates
    mu, covar = Robot.estimate(w_norm,p_m[m][tr])
    mean_estimate[m][tr].append(mu)
    if graphics:
        #arbitrarily select the first trial for graphics
        vis.visualize(Bot,t,p2,p_m[0][0],w_norm,mu)
    for tr in range(trials):
        resample_percentage[tr] = 100.0*(resample_count[tr]/time_steps)
    tr += 1
```




Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

![GIF](ParticleFilterTracking.gif)
