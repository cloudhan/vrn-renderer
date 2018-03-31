import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", required=True)
parser.add_argument("-o", default="face.obj", type=str)
parser.add_argument("--threshold", default=0.4, type=float)
args = parser.parse_args()

assert args.o.endswith(".obj")
basename = args.o.rsplit(".", 1)[0]
mask_name = basename + "_mask.png"
diffuse_name = basename + "_diffuse.png"

import warnings
warnings.simplefilter("ignore")

import mcubes
import numpy as np
import torch as th

from PIL import Image
from vrn_pytorch.vrn_unguided import vrn_unguided
from utils import chw2hwc, hwc2chw, write_mask

xsize = 192
ysize = 192

print("Loading VRN model...")
vrn_unguided.load_state_dict(th.load("./vrn_pytorch/vrn_unguided.pth"))
im = np.asarray(Image.open(args.i).resize((xsize,ysize)), dtype=np.float32)[:,:,:3]

print("Computing face volume...")
vols = vrn_unguided(th.autograd.Variable(th.Tensor(np.array([hwc2chw(im)]))))
vol = (vols[0][0].data.numpy() * 255).astype(np.uint8)

mask = (np.sum(vol, axis=0) > 1) * 255.0

# write_mask(mask_name, xsize, ysize, mask)

# vol = (v > args.threshold) * 255.0

print("Computing surface mesh...")
vertices, triangles = mcubes.marching_cubes(vol, 1)
vertices = vertices[:,(2,1,0)]
vertices[:,2] *= 0.5 # scale the Z component correctly
mcubes.export_obj(vertices, triangles, args.o)

# print("""Done.
#   mesh saved as {}
#   mask saved as {}""".format(args.o, mask_name))

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


############################################################################
############################################################################
############################################################################
xsize = 384
ysize = 384
im = np.asarray(Image.open(args.i).resize((xsize,ysize)), dtype=np.float32)[:,:,:3]
mask = np.asarray(Image.fromarray(mask).resize((xsize,ysize)), dtype=np.float32)

print("Loading IID Model...")

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import mxnet as mx
from IID.symbol import generator_symbol

args = mx.nd.load("IID/albedo_style_20180226_135333_args_001.nd")
auxs = mx.nd.load("IID/albedo_style_20180226_135333_auxs_001.nd")


ctx = mx.gpu(0)
batch_size = 1

print("Computing Albedo...")

for key in args:
    args[key] = args[key].as_in_context(ctx)
for key in auxs:
    auxs[key] = auxs[key].as_in_context(ctx)
args["image"] = mx.nd.zeros((batch_size,3,ysize,xsize), ctx)
args["mask"] = mx.nd.zeros((batch_size,1,ysize,xsize), ctx)
gene = generator_symbol().bind(ctx=ctx, args=args, aux_states=auxs, grad_req="null")

image_nd = mx.nd.array([hwc2chw(im)])
mask_nd = mx.nd.array([[mask]])

image_nd.copyto(gene.arg_dict["image"])
mask_nd.copyto(gene.arg_dict["mask"])
outputs = gene.forward()

diffuse = chw2hwc(outputs[0][0].asnumpy())
Image.fromarray(diffuse.astype(np.uint8)).save(diffuse_name)

print("Done")