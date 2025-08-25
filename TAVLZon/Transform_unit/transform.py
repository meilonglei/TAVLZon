import numpy as np
import quaternion

def habitat_translation(position):
    return np.array([position[0],position[2],position[1]])

def habitat_rotation(rotation):
    rotation_matrix = quaternion.as_rotation_matrix(rotation)
    transform_matrix = np.array([[1,0,0],
                                 [0,0,1],
                                 [0,1,0]])
    rotation_180_x = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    rotation_matrix = rotation_180_x @ np.matmul(transform_matrix,rotation_matrix)
    return rotation_matrix



    