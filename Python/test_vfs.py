import os
from pathlib import Path
import cv2
import numpy as np

import pyvfs
import vfs_params

if __name__ == '__main__':
    fpdn = Path(os.path.dirname(__file__)).parent
    fpdn_image = os.path.join(fpdn, 'test_vfs')

    # read images
    im1 = cv2.imread(os.path.join(fpdn_image, 'robot1.png'), cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(os.path.join(fpdn_image, 'robot2.png'), cv2.IMREAD_GRAYSCALE)

    p = vfs_params.vfs_params()
    vfs = pyvfs.vfs()
    ret = vfs.initialize(p)

    # read vector field
    translationVector = cv2.readOpticalFlow(os.path.join(fpdn_image, 'translationVector.flo'))
    calibrationVector = cv2.readOpticalFlow(os.path.join(fpdn_image, 'calibrationVector.flo'))
    # read fisheye mask
    fisheyeMask8 = cv2.imread(os.path.join(fpdn_image, 'mask.png'), cv2.IMREAD_GRAYSCALE)
    fisheyeMask = fisheyeMask8.astype(np.float32)/255
    ret = vfs.copy_mask_to_device(fisheyeMask)
    ret = vfs.load_vector_fields(translationVector, calibrationVector)

    equi1 = cv2.equalizeHist(im1)
    equi2 = cv2.equalizeHist(im2)
    ret = vfs.copy_images_to_device(equi1, equi2)
    ret = vfs.solve_stereo_forward_masked()

    p_kb = vfs_params.kb_params()
    depth, X, disparity, uvrgb = vfs.copy_result_to_host(p_kb)

    uvrgb8 = (uvrgb * 255).astype(np.uint8)
    cv2.writeOpticalFlow("disparity.flo", disparity)
    cv2.imwrite("request_uvrgb8.png", uvrgb8)
    cv2.imwrite("request_X.png", X*20.0)
    cv2.imwrite("request_depth.png", depth*20.0)
