# Dual Quaternions

## a library reponsible for converting 4x4 Matrices to/from dual-quaternions

I don't own this library, it was converted from its javascript source that can be found here:

[mat4-to-dual-quat.js](https://github.com/chinedufn/mat4-to-dual-quat/blob/master/src/mat4-to-dual-quat.js)

[dual-quat-to-mat4.js](https://github.com/chinedufn/dual-quat-to-mat4/blob/master/src/dual-quat-to-mat4.js)

also, both of those files depend on the library [gl-quat](https://github.com/stackgl/gl-quat) and [gl-vec4](https://github.com/stackgl/gl-vec4)

### all functions and their original counterparts

**function name**|**link to source**
:-----|:-----
convertMatrixToDualQuat|[mat4-to-dual-quat.js](https://github.com/chinedufn/mat4-to-dual-quat/blob/master/src/mat4-to-dual-quat.js)
convertDualQuatToMatrix|[dual-quat-to-mat4.js](https://github.com/chinedufn/dual-quat-to-mat4/blob/master/src/dual-quat-to-mat4.js)
quatFromMat3|[fromMat3.js](https://github.com/stackgl/gl-quat/blob/master/fromMat3.js)
quatMultiply|[multiply.js](https://github.com/stackgl/gl-quat/blob/master/multiply.js)
quatScale|[scale.js](https://github.com/stackgl/gl-vec4/blob/master/scale.js)

again, all credit goes to [Chinedu Francis Nwafili](https://github.com/chinedufn) \
contact: frankie.nwafili@gmail.com