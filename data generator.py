# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

datapoints = 10000
min_data = 0
max_data = 50
noise_stds = [0,0.001,0.01,0.1,0.2,0.5]
trial_nums = 30

ts = np.linspace(min_data, max_data, datapoints)

def equation(t):
    #eq being used is x'' + x' + 0.1x = 0
    
    c = 1
    k = 0.1
    m=1
    gamma = c/(2*m)
    A = 1
    B =1
    omega = np.sqrt(k/m * gamma**2)
    
    x = np.exp(-gamma*t)*(A*np.cos(omega*t)+ B*np.sin(omega*t))
    return x

for noise_std in noise_stds:
    for trial_ID in range(trial_nums):
        seed = trial_ID
        rng = np.random.default_rng(seed)
        
        
        
        errors = rng.normal(0, noise_std, datapoints)
        
        # generating xs
        xs = equation(ts)
        actual_xs = xs + errors
        
        data = []
        
        for t, x in zip(ts, actual_xs):
            
            data.append({'noise_std': noise_std, 'trial_ID': trial_ID, 'time': t, 'displacement': x})
    
        df = pd.DataFrame(data)
        df.to_csv(f'generated_data\oscillator_data__noise{noise_std}__trial_ID{trial_ID}.csv', index=False)