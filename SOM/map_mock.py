import tensorflow as tf
from math import *
import sys
sys.path.append("../TOOLBOX")
from map_som import *

from tool_norm import *


res_path = "../RES/"

w = np.load(res_path+"w_ComaBer_85target_pm1_dn.npy")
flag = 0
map_oc = MAP_SOM(w,flag)


inputs_oc= np.load(res_path+"inputs_xyzpmradec_ComaBer_85target_pm1_dn.npy")

inputs_mock = np.load(res_path+"inputs_xyzpmradec_ComaBer_mock.npy")
inputs = normalization_2sigma(inputs_mock,inputs_oc)

print (inputs.shape)

map_mock, d_vw_mock = map_oc.map_vects(inputs)

np.save(res_path + "map_mock_ComaBer_85target_pm1_dn.npy",map_mock)
np.save(res_path + "d_vw_mock_ComaBer_85target_pm1_dn.npy",d_vw_mock)



