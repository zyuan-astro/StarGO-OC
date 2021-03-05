import tensorflow as tf
import numpy as np
import os
from math import *
from som_tool import * 

os.environ["CUDA_VISIBLE_DEVICES"]=""


class MAP_SOM(object):

    
    
    def __init__(self, weightage_vects, flag):
        
        
        self.weightage_vects = weightage_vects
        m = self.weightage_vects.shape[0]
        n = self.weightage_vects.shape[1]
        dim = self.weightage_vects.shape[2]
        
        self.flag = flag
       
        self._graph = tf.Graph()
       
    
        with self._graph.as_default():
        
           
            self.a = tf.range(0, m, 1, dtype = tf.int64) 
            self.b = tf.range(0, n, 1, dtype = tf.int64) 
            self.ones = tf.constant(1., shape = [m, n, 1])
                
            grid_x, grid_y = tf.meshgrid(self.a, self.b)
            zeros_vects= tf.Variable(tf.zeros([m, n, dim]))
            
            self.grid_xy = tf.stack([grid_y, grid_x], axis = 2)
            
          
            self.vect_input = tf.placeholder("float", dim)
            
            with tf.name_scope('bmu'):

                self.dist_vects = self.vect_input - self.weightage_vects

                if self.flag == 1:
                    
                    
                    
                    input_vects = self.vect_input - zeros_vects
                    
                    
              
                    lon1 = tf.slice(input_vects, [0, 0, dim-1], [m, n, 1]) * 2 * pi
                    lon2 = tf.slice(self.weightage_vects, [0, 0, dim-1], [m, n, 1]) * 2 * pi
            
                    #latitude must be in (-pi/2, pi/2)#
                    lat1 = (tf.slice(input_vects, [0, 0, dim-2], [m, n, 1]) - 0.5) * pi
                    lat2 = (tf.slice(self.weightage_vects, [0, 0, dim-2], [m, n, 1]) -0.5) * pi
                    
                    #Normalize Great Circle Distance py pi#
                    dist_gc = cal_d_gc(lon1, lon2, lat1, lat2) / pi
                     
                    '''Redefine distance vectors'''
                    
                    dw01 = tf.slice(self.dist_vects , [0, 0, 0], [m, n, 2])
                     
                    self.dist_vects = tf.concat([dw01, dist_gc], -1)
                
                '''Calculate distance scalors'''
                  
                self.dist = tf.sqrt(tf.reduce_sum(tf.pow(self.dist_vects, 2), 2))
            
                    
                ''' Get Index of BMU'''
           
                ind_col = tf.argmin(self.dist, 1)
                index_col = tf.stack([self.a, ind_col], axis = 1)
                d_col = tf.gather_nd(self.dist, index_col)
                
                ind_row = tf.argmin(d_col, 0)
                self.bmu_index = tf.gather(index_col, ind_row)
            
                
                
            ##INITIALIZE SESSION
            self._sess = tf.Session()
            
            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    
   
        
       
    def map_vects(self, input_vects):
        
        map_bmu = []
        d_vw = []
        for input_vect in input_vects:
            
            
            bmu_index = self._sess.run([self.bmu_index], feed_dict={self.vect_input: input_vect})
            d = self._sess.run([self.dist], feed_dict={self.vect_input: input_vect})
            
            d0 = np.array(d)[0]
            bmu0 = np.array(bmu_index)[0]
            
            
            map_bmu = np.append(map_bmu, bmu_index)    
            d_vw = np.append(d_vw, d0[bmu0[0],bmu0[1]])    
            
         
        map_bmu = np.reshape(np.int32(map_bmu), [len(input_vects), 2])
        
        
        return map_bmu, d_vw
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    