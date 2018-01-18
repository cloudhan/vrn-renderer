import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", required=True)
args = parser.parse_args()

import numpy as np
import torch as th
import matplotlib.pyplot as plt

from PIL import Image
from vrn_pytorch.vrn_unguided import vrn_unguided


def chw2hwc(data):
    """Rollaxis of data from channel X height X width (CHW) to HWC layout"""
    assert data.shape[0] <= 4, "Image is not CHW layout"
    return np.rollaxis(data, 0, 3)

def hwc2chw(data):
    """Rollaxis of data from height X width X channel (HWC) to CHW layout"""
    assert data.shape[2] <= 4, "Image is not HWC layout"
    return np.rollaxis(data, 2, 0)


vrn_unguided.load_state_dict(th.load("./vrn_pytorch/vrn_unguided.pth"))

im = np.asarray(Image.open(args.i).resize((192,192)), dtype=np.float32)


vols = vrn_unguided(th.autograd.Variable(th.Tensor(np.array([hwc2chw(im)]))))
vol = vols[0][0].data.numpy() * 255

import visvis as vv
t = vv.imshow(im)
t.interpolate = True # interpolate pixels

# volshow will use volshow3 and rendering the isosurface if OpenGL
# version is >= 2.0. Otherwise, it will show slices with bars that you
# can move (much less useful).
volRGB = np.stack(((vol > 1) * im[:,:,0],
                   (vol > 1) * im[:,:,1],
                   (vol > 1) * im[:,:,2]), axis=3)
print volRGB.shape

v = vv.volshow(volRGB, renderStyle='iso')
v.transformations[1].sz = 0.5

l0 = vv.gca()
l0.light0.ambient = 0.9 # 0.2 is default for light 0
l0.light0.diffuse = 1.0 # 1.0 is default

a = vv.gca()
a.camera.fov = 0 # orthographic

vv.use().Run()
