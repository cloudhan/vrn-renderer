import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", required=True)
parser.add_argument("-o", default="face.obj", type=str)
parser.add_argument("--threshold", default=0.4, type=float)
args = parser.parse_args()

assert args.o.endswith(".obj")

import mcubes
import numpy as np
import torch as th

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


print("Loading VRN model...")
vrn_unguided.load_state_dict(th.load("./vrn_pytorch/vrn_unguided.pth"))
im = np.asarray(Image.open(args.i).resize((192,192)), dtype=np.float32)

print("Computing face volume...")
vols = vrn_unguided(th.autograd.Variable(th.Tensor(np.array([hwc2chw(im)]))))
vol = (vols[0][0].data.numpy() > args.threshold) * 255.0

print("Computing surface mesh...")
vertices, triangles = mcubes.marching_cubes(vol, 1)
vertices = vertices[:,(2,1,0)]
vertices[:,2] *= 0.5 # scale the Z component correctly
mcubes.export_obj(vertices, triangles, args.o)

## Uncomment following code to generate face object with per-vertex color
## Useless in our application
# import scipy.misc
# from sklearn.neighbors import NearestNeighbors

# r = im[:,:,0].flatten()
# g = im[:,:,1].flatten()
# b = im[:,:,2].flatten()

# vcx,vcy = np.meshgrid(np.arange(0,192),np.arange(0,192))
# vcx = vcx.flatten()
# vcy = vcy.flatten()
# vc = np.vstack((vcx, vcy, r, g, b)).transpose()
# neigh = NearestNeighbors(n_neighbors=1)
# neigh.fit(vc[:,:2])
# n = neigh.kneighbors(vertices[:,(0,1)], return_distance=False)
# colour = vc[n,2:].reshape((vertices.shape[0],3)).astype(float) / 255

# vc = np.hstack((vertices, colour))

# with open(args.o, 'w') as f:
#     for v in range(0,vc.shape[0]):
#         f.write('v %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n' % (vc[v,0],vc[v,1],vc[v,2],vc[v,3],vc[v,4],vc[v,5]))

#     for t in range(0,triangles.shape[0]):
#         f.write('f {} {} {}\n'.format(*triangles[t,:]+1))

# print('Calculated the isosurface.')
