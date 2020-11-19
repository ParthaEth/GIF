import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

from time import time
from . import standard_rasterize_cuda

def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def get_visibility(vertices, triangles, h, w, print_time = False):
    '''
        vertices: [batch_size, nv, 3]. range:[-1, 1]
        triangles: [batch_size, nf, 3]
    '''
    bz = vertices.shape[0]
    device = vertices.device
    # vertices[...,-1] = -vertices[...,-1]
    vertices = vertices.clone()
    vertices[...,0] = vertices[..., 0]*w/2 + w/2 
    vertices[...,1] = vertices[..., 1]*h/2 + h/2 
    vertices[...,2] = vertices[..., 2] - vertices[..., 2].min() + 1

    depth_buffer = torch.zeros([bz, h, w]).float().to(device) + 1e6
    triangle_buffer = torch.zeros([bz, h, w]).int().to(device) - 1
    baryw_buffer = torch.zeros([bz, h, w, 3]).float().to(device)
    vert_vis = torch.zeros([bz, vertices.shape[1]]).float().to(device)
    
    st = time()
    f_vs = face_vertices(vertices, triangles)
    ## before standaard_rasterize_cuda
    # make sure f_vs is in image space, z is larger than 0
    standard_rasterize_cuda.standard_rasterize(f_vs, depth_buffer, triangle_buffer, baryw_buffer, h, w)

    triangle_buffer = triangle_buffer.reshape(bz, -1)
    for i in range(bz):
        tri_visind = torch.unique(triangle_buffer[i])[1:].long()
        vert_visind = triangles[i,tri_visind,:].flatten()
        vert_vis[i, torch.unique(vert_visind.long())] = 1.0
    if print_time:
        print(time() - st)
    return vert_vis

def get_visibility_z(vertices, triangles, h, w, print_time = False):
    '''
        vertices: [batch_size, nv, 3]. range:[-1, 1]
        triangles: [batch_size, nf, 3]
    '''
    bz = vertices.shape[0]
    device = vertices.device
    vertices = vertices.clone()
    vertices[...,0] = vertices[..., 0]*w/2 + w/2 
    vertices[...,1] = vertices[..., 1]*h/2 + h/2 
    vertices[...,2] = vertices[..., 2] - vertices[..., 2].min() + 1

    depth_buffer = torch.zeros([bz, h, w]).float().to(device) + 1e6
    triangle_buffer = torch.zeros([bz, h, w]).int().to(device) - 1
    baryw_buffer = torch.zeros([bz, h, w, 3]).float().to(device)
    vert_vis = torch.zeros([bz, vertices.shape[1]]).float().to(device)
    
    st = time()
    f_vs = face_vertices(vertices, triangles)
    ## before standaard_rasterize_cuda
    # make sure f_vs is in image space, z is larger than 0
    standard_rasterize_cuda.standard_rasterize(f_vs, depth_buffer, triangle_buffer, baryw_buffer, h, w)

    zrange = vertices[...,-1].max() - vertices[...,-1].min()
    for i in range(bz):
        for j in range(vertices.shape[1]):
            [x,y,z] = vertices[i, j]
            ul = depth_buffer[i, int(torch.floor(y)), int(torch.floor(x))]
            ur = depth_buffer[i, int(torch.floor(y)), int(torch.ceil(x))]
            dl = depth_buffer[i, int(torch.ceil(y)), int(torch.floor(x))]
            dr = depth_buffer[i, int(torch.ceil(y)), int(torch.ceil(x))]

            yd = y - torch.floor(y)
            xd = x - torch.floor(x)
            depth = ul*(1-xd)*(1-yd) + ur*xd*(1-yd) + dl*(1-xd)*yd + dr*xd*yd
            if z < depth + zrange*0.02:
                vert_vis[i, j] = 1.0
    print(time() - st)
    return vert_vis
