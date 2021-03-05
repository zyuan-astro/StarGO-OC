import tensorflow as tf
from math import *
import numpy as np
import sys
import os

sys.path.append("../TOOLBOX/")
from som import *
from tool_norm import *


res_path="../RES/"

inputs0 = np.load(res_path+"inputs_xyzpmradec_ComaBer_85target_pm1_dn.npy")


inputs = normalization_2sigma(inputs0, inputs0)


m = 100
n = 100
dim = inputs.shape[1]
print (inputs.shape)
n_intr = 400


flag = 0

som = SOM(m, n, dim, n_intr, flag)


som.train(inputs)


w = som.get_weights()
u = som.cal_u_matrix()

np.save(res_path+"w_ComaBer_85target_pm1_dn.npy", w)

np.save(res_path+"u_ComaBer_85target_pm1_dn.npy", u)



map_oc, d_vw_oc = som.map_vects(inputs)



np.save(res_path+"map_ComaBer_85target_pm1_dn.npy", map_oc)
np.save(res_path+"d_vw_ComaBer_85target_pm1_dn.npy", d_vw_oc)

