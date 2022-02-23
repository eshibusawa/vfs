# BSD 2-Clause License
#
# Copyright (c) 2022, Eijiro SHIBUSAWA
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
