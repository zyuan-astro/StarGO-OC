import numpy as np
from math import *


def normalization_2sigma(inputs, inputs0):

    inputs_norm = np.empty_like(inputs)
    
    for i in range(inputs.shape[1]):
        
        norm_range = np.percentile(inputs0[:,i],97.5)-np.percentile(inputs0[:,i],2.5)
            
        inputs_norm[:,i] = (inputs[:,i]-np.percentile(inputs0[:,i],2.5))/norm_range
            
   
    return inputs_norm




def normalization_angle_2sigma(inputs, inputs0):

    inputs_norm = np.empty_like(inputs)
    
    for i in range(inputs.shape[1]):
        
        if i < 2:
         
            norm_range = np.percentile(inputs0[:,i],97.5)-np.percentile(inputs0[:,i],2.5)
            
            inputs_norm[:,i] = (inputs[:,i]-np.percentile(inputs0[:,i],2.5))/norm_range
            
        elif i == 2 :
            
            
            inputs_norm[:,2] = inputs[:,2]/180
            
        else:
            
            inputs_norm[:,3] = inputs[:,3]/360 
   
    return inputs_norm


def normalization_angle_1sigma(inputs, inputs0):

    inputs_norm = np.empty_like(inputs)
    
    for i in range(inputs.shape[1]):
        
        if i < 2:
         
            norm_range = np.percentile(inputs0[:,i],84)-np.percentile(inputs0[:,i],16)
            
            inputs_norm[:,i] = (inputs[:,i]-np.percentile(inputs0[:,i],16))/norm_range
            
        elif i == 2 :
            
            
            inputs_norm[:,2] = inputs[:,2]/180
            
        else:
            
            inputs_norm[:,3] = inputs[:,3]/360 
   
    return inputs_norm


