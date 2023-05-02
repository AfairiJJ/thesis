import numpy as np

def poisson_deviance(pred, obs):
    return 200*(sum(pred)-sum(obs)+sum(np.log((obs/pred)**(obs)))) / len(pred)