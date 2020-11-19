"""
Demo visibility.
"""

from helpers import *
from visibility import *
#import soft_renderer.cuda.standard_rasterize as standard_rasterize_cuda

def main():
    # load obj
    mesh = Mesh.from_obj('./data/obj/body.obj', normalization=False, load_texture=False, texture_type='surface')
    vertices = mesh.vertices*0.8
    print(mesh.vertices)
    print(mesh.vertices.shape)
    print('____________')
    triangles = mesh.faces
    print(mesh.faces)
    print(mesh.faces.shape)
    print('____________')
    # vertices = vertices.expand(100,-1,-1)
    # triangles = triangles.expand(100,-1,-1)
    vert_vis = get_visibility(vertices, triangles, h=512, w =512, print_time = True) # vertices: projected vertices in image coordinate: u /in [-1,1], v /in [-1, 1], z in [-inf, inf]

    ## save obj
    vertices = vertices.detach().cpu().numpy()[0]
    triangles = triangles.detach().cpu().numpy()[0]
    vert_vis = vert_vis.detach().cpu().numpy()[0]
    colors = np.tile(vert_vis[:,None], (1,3))

    obj_name = './data/obj/body_vis.obj'
    print('saving obj...')
    write_obj_with_colors(obj_name, vertices, triangles, colors)
    print('Done! Check obj in {}'.format(obj_name))
    



if __name__ == '__main__':
    main()
