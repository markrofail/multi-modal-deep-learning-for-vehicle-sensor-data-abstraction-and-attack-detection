# import numpy as np
# # from helpers import quaternion
# from helpers.quaternion.helpers import (mat3_mat4, vec_mat4)


# def test_mat3_mat4():
#     mat3 = np.identity(3)
#     mat4 = np.identity(4)

#     assert np.array_equal(mat3, mat3_mat4(mat4)),\
#         'The function did not extract the 3x3 Matrix correctly'


# def test_vec_mat4():
#     vec = np.append(np.arange(3), [0])
#     mat4 = np.identity(4)
#     mat4[:][3] = vec
#     mat4 = mat4.T

#     assert np.array_equal(vec, vec_mat4(mat4)),\
#         'The function did not extract the Vector correctly'


# from helpers.quaternion.quat_op import (quat_mult, quat_scale)
# def test_quat_mult():
#     a = np.array([2, 3, 4, 3])
#     b = np.array([4, 3.9, -1, -3])
#     ans = np.array([9.3, 10.8, 34.7, -12.6])

#     mult = np.array(quat_mult(a, b))

#     print(mult)
#     assert np.array_equal(mult, ans),\
#         'The function did not compute the product correctly'
