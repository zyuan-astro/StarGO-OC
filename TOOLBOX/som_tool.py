import tensorflow as tf
import numpy as np
import os

#os.environ["CUDA_VISIBLE_DEVICES"]=""


    
    
def cal_d_gc(lon1, lon2, lat1, lat2):
    
                  
    d_lon = tf.abs(tf.subtract(lon1, lon2))                  
    d_lat = tf.abs(tf.subtract(lat1, lat2))
                                        
    term1 = tf.pow(tf.sin(d_lat/2), 2.)
    term2 = tf.multiply(tf.multiply(tf.cos(lat1), tf.cos(lat2)), tf.pow(tf.sin(d_lon/2.),2))
    
    f = tf.sqrt(tf.add(term1,term2))
                
    dist_gc = 2 * tf.asin(f) 
                     
    return dist_gc
    
