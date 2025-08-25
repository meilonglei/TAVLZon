import numpy as np
import quaternion

def habitat_camera_extrinsic(position,rotation):
   
    extrinsic = np.eye(4)
    extrinsic[0:3,0:3] = rotation 
    extrinsic[0:3,3] = position

    return extrinsic
