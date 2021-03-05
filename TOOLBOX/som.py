import tensorflow as tf
import numpy as np
import os
from som_tool import * 
from tool import * 

#os.environ["CUDA_VISIBLE_DEVICES"]=""


class SOM(object):

    
    
    def __init__(self, m, n, dim, n_iterations, flag, alpha=None, sigma=None):
        
        
        
        self.m = m
        self.n = n
        self.dim = dim 
        self.alpha = 0.3        
        self.sigma = max(m, n) / 2.0
        self._graph = tf.Graph()
        self.n_iterations = n_iterations
        self.flag = flag
    
    
        with self._graph.as_default():
        
           
            self.a = tf.range(0, m, 1, dtype = tf.int64) 
            self.b = tf.range(0, n, 1, dtype = tf.int64) 
            self.ones = tf.constant(1., shape = [m, n, 1])
                
            grid_x, grid_y = tf.meshgrid(self.a, self.b)
            zeros_vects= tf.Variable(tf.zeros([m, n, dim]))
            self.grid_xy = tf.stack([grid_y, grid_x], axis = 2)
            

            self.weightage_vects= tf.Variable(tf.random.uniform([m, n, dim]))
            
                
            
            self.vect_input = tf.compat.v1.placeholder("float", [dim])
            self.iter_input = tf.compat.v1.placeholder("float")
            self.dist_vects_cart = self.vect_input - self.weightage_vects

            
            with tf.name_scope('bmu'):

                self.dist_vects = self.vect_input - self.weightage_vects
                
                if self.flag > 0:
                   
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
                
                ''' Calculate distance scalors '''
                  
                self.dist = tf.sqrt(tf.reduce_sum(tf.pow(self.dist_vects, 2), 2))
            
                    
                ''' Get Index of BMU'''
           
                ind_col = tf.argmin(self.dist, 1)
                index_col = tf.stack([self.a, ind_col], axis = 1)
                d_col = tf.gather_nd(self.dist, index_col)
                
                ind_row = tf.argmin(d_col, 0)
                self.bmu_index = tf.gather(index_col, ind_row)
                
                ''' Get the distance between neurons and BMU'''
          
                d_bmu = tf.reduce_sum(tf.pow(tf.subtract(self.grid_xy, self.bmu_index), 2), 2)

            with tf.name_scope('learning_rate'):    

                learning_rate_op = tf.subtract(1.0, tf.math.divide(self.iter_input, self.n_iterations))
                alpha_op = tf.multiply(self.alpha, learning_rate_op)
                sigma_op = tf.multiply(self.sigma, learning_rate_op)
        
                func = tf.exp(tf.negative(tf.math.divide(tf.cast(d_bmu, "float32"), tf.pow(sigma_op, 2))))
                learning_rate_op = tf.multiply(alpha_op, func)
                
            with tf.name_scope('dw'):
    
                factor = tf.stack([learning_rate_op for i in range(dim)], axis = -1)
        
                ''' Update the weights in Cartesian Coordinate '''
        
                dw = tf.multiply(factor, self.dist_vects_cart)  
  
            with tf.name_scope("train"):

                
                w_op = tf.add(self.weightage_vects, dw)
                self.training_op = tf.compat.v1.assign(self.weightage_vects, w_op)   
            
          
            ##INITIALIZE SESSION
            self._sess = tf.compat.v1.Session()
            
            ##INITIALIZE VARIABLES
            init_op = tf.compat.v1.global_variables_initializer()
            self._sess.run(init_op)

    
    
    def train(self, input_vects):
 
        
        
        for iter_no in range(self.n_iterations):
            
            
          
            print (iter_no)
            
           
            for input_vect in input_vects:
               
                self._sess.run([self.training_op], feed_dict={self.vect_input: input_vect, self.iter_input: iter_no})                       
   
    def get_weights(self):
        
        self.w0 = self._sess.run(self.weightage_vects)
        
        return self.w0
    
    
    
    def cal_u_matrix(self):
        
        
        if self.flag > 0:
                    
            u = re_cal_u_angle(self.w0)
           
        else:
            
            dw_row = np.sum(np.power(np.diff(self.w0, axis = 0), 2), axis = -1)
            dw_col = np.sum(np.power(np.diff(self.w0, axis = 1), 2), axis = -1)
            u = dw_row[:, :-1] + dw_col[:-1, :]
        
        return u
    
    
    
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
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
