import numpy as np




class GR(object):
   
    
    def __init__(self, u_matrix, u_tre):

        self.u = u_matrix
        u_grp = np.where(self.u <= u_tre)
        u_bnd = np.where(self.u > u_tre)
        # 2-D neuron map: m*n
        
        self.m = u_matrix.shape[0] + 1
        self.n = u_matrix.shape[1] + 1
        
        x = np.arange(0, self.m, 1)
        y = np.arange(0, self.n, 1)
        
        x_bnd = x[u_bnd[0]]
        y_bnd = y[u_bnd[1]]
        
        x_grp0 = x[u_grp[0]]
        y_grp0 = y[u_grp[1]]
        
        
        ind_b1 = (x_grp0 == (self.m-2))
        
        y_b1 = y_grp0[ind_b1]
        x_b1 = np.full_like(y_b1, (self.m-1))
        
        
        ind_b2 = (y_grp0 == (self.n-2))
        
        x_b2 = x_grp0[ind_b2]
        y_b2 = np.full_like(x_b2, (self.n-1))
        
        
        x_grp = np.append(x_grp0, x_b1)
        y_grp = np.append(y_grp0, y_b1)
        
        x_grp = np.append(x_grp, x_b2)
        y_grp = np.append(y_grp, y_b2)

       
        self.grp = np.empty([len(x_grp), 2])
        
        
        self.grp[:, 0] = x_grp
        self.grp[:, 1] = y_grp
        
        self.bnd = np.empty([len(x_bnd), 2])
        
        self.bnd[:, 0] = x_bnd
        self.bnd[:, 1] = y_bnd
        
        self.id_grp = np.int64(x_grp * self.m + y_grp)
        
    def get_xy_grp(self):
        
        return self.grp
    
    def get_xy_bnd(self):
        
        return self.bnd
        
    def get_id_grp(self):
        
        return self.id_grp
    
    
    def get_group(self, seed):
        
        
         
        grp0 = np.atleast_1d(seed)
        
        n0 = 0
            
        while len(grp0) > n0:
            
            n0 = len(grp0)
            for a in (-1, 1):
                
                for b in (-1, 1):
                    
                    grp0 = grow_region(seed, self.id_grp, grp0, self.m, self.n, a, b)
                    
        return grp0
        
        
def grow_region(seed, id_grp, grp0, m, n, a, b):
        
   
        
    x0 = np.int64(seed / m)
    y0 = seed - x0 * m
        
        
    while ((y0 < n) * (y0 >= 0)):
    
        x0 = np.int64(seed / m)
        
        while ((x0 < m) * (x0 >= 0)):
 
            e0 = x0 * m + y0
            ind = np.in1d(id_grp, e0)
    
            if np.any(ind):
        
                grp0_x = np.int64(grp0 / m)
                grp0_y = grp0 - grp0_x * m
    
                d = (grp0_x - x0)**2.0 + (grp0_y - y0)**2.0 
        
                if min(d) == 1:
        
                    grp0 = np.append(grp0, e0)
            
                
            
            x0 += a 
        
        y0 += b
       
    return grp0
     
    