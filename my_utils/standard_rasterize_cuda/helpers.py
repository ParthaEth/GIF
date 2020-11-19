import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def save_obj(filename, vertices, faces, textures=None, uvcoords=None, uvfaces=None, texture_type='surface'):
    assert vertices.ndimension() == 2
    assert faces.ndimension() == 2
    assert texture_type in ['surface', 'vertex']
    # assert texture_res >= 2

    if textures is not None and texture_type == 'surface':
        textures =textures.detach().cpu().numpy().transpose(1,2,0)
        filename_mtl = filename[:-4] + '.mtl'
        filename_texture = filename[:-4] + '.png'
        material_name = 'material_1'
        # texture_image, vertices_textures = create_texture_image(textures, texture_res)
        texture_image = textures
        texture_image = texture_image.clip(0, 1)
        texture_image = (texture_image * 255).astype('uint8')
        imsave(filename_texture, texture_image)

    faces = faces.detach().cpu().numpy()

    with open(filename, 'w') as f:
        f.write('# %s\n' % os.path.basename(filename))
        f.write('#\n')
        f.write('\n')

        if textures is not None:
            f.write('mtllib %s\n\n' % os.path.basename(filename_mtl))

        if textures is not None and texture_type == 'vertex':
            for vertex, color in zip(vertices, textures):
                f.write('v %.8f %.8f %.8f %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2], 
                                                               color[0], color[1], color[2]))
            f.write('\n')
        else:
            for vertex in vertices:
                f.write('v %.8f %.8f %.8f\n' % (vertex[0], vertex[1], vertex[2]))
            f.write('\n')

        if textures is not None and texture_type == 'surface':
            for vertex in uvcoords.reshape((-1, 2)):
                f.write('vt %.8f %.8f\n' % (vertex[0], vertex[1]))
            f.write('\n')

            f.write('usemtl %s\n' % material_name)
            for i, face in enumerate(faces):
                f.write('f %d/%d %d/%d %d/%d\n' % (
                    face[0] + 1, uvfaces[i,0]+1, face[1] + 1, uvfaces[i,1]+1, face[2] + 1, uvfaces[i,2]+1))
            f.write('\n')
        else:
            for face in faces:
                f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))

    if textures is not None and texture_type == 'surface':
        with open(filename_mtl, 'w') as f:
            f.write('newmtl %s\n' % material_name)
            f.write('map_Kd %s\n' % os.path.basename(filename_texture))

def load_obj(filename_obj, normalization=False, load_texture=False, texture_res=4, texture_type='surface'):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    assert texture_type in ['surface', 'vertex']

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = torch.from_numpy(np.vstack(vertices).astype(np.float32)).cuda()

    if load_texture: 
        print("Loading texture function is not implemented!") 
        exit()
        # uvfaces = []
        # uvcoords = []; 
        # for line in lines:
        #     if len(line.split()) == 0:
        #         continue
        #     if line.split()[0] == 'vt':
        #         uvcoords.append([float(v) for v in line.split()[1:3]])
        #     if line.split()[0] == 'f':
        #         vs = line.split()[1:]
        #         nv = len(vs)
        #         v0 = int(vs[0].split('/')[1])
        #         for i in range(nv - 2):
        #             v1 = int(vs[i + 1].split('/')[1])
        #             v2 = int(vs[i + 2].split('/')[1])
        #             uvfaces.append((v0, v1, v2))
        # uvcoords = torch.from_numpy(np.vstack(uvcoords).astype(np.float32)).cuda()
        # uvfaces = torch.from_numpy(np.vstack(uvfaces).astype(np.int32)).cuda() - 1

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = torch.from_numpy(np.vstack(faces).astype(np.int32)).cuda() - 1
    
    # load textures
    if load_texture and texture_type == 'surface':
        print("Loading texture function is not implemented!") 
        exit()
        # textures = None
        # for line in lines:
        #     if line.startswith('mtllib'):
        #         filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
        #         textures = load_textures(filename_obj, filename_mtl, texture_res)
        # if textures is None:
        #     raise Exception('Failed to load textures.')
    elif load_texture and texture_type == 'vertex':
        print("Loading texture function is not implemented!") 
        exit()
        # textures = []
        # for line in lines:
        #     if len(line.split()) == 0:
        #         continue
        #     if line.split()[0] == 'v':
        #         textures.append([float(v) for v in line.split()[4:7]])
        # textures = torch.from_numpy(np.vstack(textures).astype(np.float32)).cuda()

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= torch.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    if load_texture:
        print("Loading texture function is not implemented!") 
        exit()
        # return vertices, faces, textures, uvcoords, uvfaces
    else:
        return vertices, faces
         
class Mesh(object):
    '''
    A simple class for creating and manipulating trimesh objects
    '''
    def __init__(self, vertices, faces, textures=None, uvcoords=None, uvfaces=None, texture_res=1, texture_type='vertex'):
        '''
        vertices, faces and textures(if not None) are expected to be Tensor objects
        '''
        self._vertices = vertices
        self._faces = faces
        self._uvcoords = uvcoords
        self._uvfaces = uvfaces

        if isinstance(self._vertices, np.ndarray):
            self._vertices = torch.from_numpy(self._vertices).float().cuda()
        if isinstance(self._faces, np.ndarray):
            self._faces = torch.from_numpy(self._faces).int().cuda()
        if self._vertices.ndimension() == 2:
            self._vertices = self._vertices[None, :, :]
        if self._faces.ndimension() == 2:
            self._faces = self._faces[None, :, :]

        if uvcoords is not None:
            if isinstance(self._uvcoords, np.ndarray):
                self._uvcoords = torch.from_numpy(self._uvcoords).float().cuda()
            if self._uvcoords.ndimension() == 2:
                self._uvcoords = self._uvcoords[None, :, :]
        if uvfaces is not None:
            if isinstance(self._uvfaces, np.ndarray):
                self._uvfaces = torch.from_numpy(self._uvfaces).int().cuda()
            if self._uvfaces.ndimension() == 2:
                self._uvfaces = self._uvfaces[None, :, :]

        self.device = self._vertices.device
        self.texture_type = texture_type

        self.batch_size = self._vertices.shape[0]
        self.num_vertices = self._vertices.shape[1]
        self.num_faces = self._faces.shape[1]

        self._face_vertices = None
        self._face_vertices_update = True
        self._surface_normals = None
        self._surface_normals_update = True
        self._vertex_normals = None
        self._vertex_normals_update = True

        self._fill_back = False

        # create textures
        if textures is None:
            if texture_type == 'surface':
                self._textures = torch.ones(self.batch_size, self.num_faces, texture_res**2, 3, 
                                            dtype=torch.float32).to(self.device)
                self.texture_res = texture_res
            elif texture_type == 'vertex':
                self._textures = torch.ones(self.batch_size, self.num_vertices, 3, 
                                            dtype=torch.float32).to(self.device)
                self.texture_res = 1
        else:
            if isinstance(textures, np.ndarray):
                textures = torch.from_numpy(textures).float().cuda()
            if textures.ndimension() == 3 and texture_type == 'surface':
                textures = textures[None, :, :, :]
            if textures.ndimension() == 2 and texture_type == 'vertex':
                textures = textures[None, :, :]
            self._textures = textures
            self.texture_res = int(np.sqrt(self._textures.shape[2]))

        self._origin_vertices = self._vertices
        self._origin_faces = self._faces
        self._origin_textures = self._textures

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces):
        # need check tensor
        self._faces = faces
        self.num_faces = self._faces.shape[1]
        self._face_vertices_update = True
        self._surface_normals_update = True
        self._vertex_normals_update = True

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        # need check tensor
        self._vertices = vertices
        self.num_vertices = self._vertices.shape[1]
        self._face_vertices_update = True
        self._surface_normals_update = True
        self._vertex_normals_update = True

    @property
    def textures(self):
        return self._textures

    @textures.setter
    def textures(self, textures):
        # need check tensor
        self._textures = textures

    @property
    def uvcoords(self):
        return self._uvcoords
    @uvcoords.setter
    def uvcoords(self, uvcoords):
        self._uvcoords = uvcoords

    @property
    def uvfaces(self):
        return self._uvfaces

    @property
    def face_vertices(self):
        if self._face_vertices_update:
            self._face_vertices = face_vertices(self.vertices, self.faces)
            self._face_vertices_update = False
        return self._face_vertices

    @property
    def surface_normals(self):
        if self._surface_normals_update:
            v10 = self.face_vertices[:, :, 0] - self.face_vertices[:, :, 1]
            v12 = self.face_vertices[:, :, 2] - self.face_vertices[:, :, 1]
            self._surface_normals = F.normalize(torch.cross(v12, v10), p=2, dim=2, eps=1e-6)
            self._surface_normals_update = False
        return self._surface_normals

    @property
    def vertex_normals(self):
        if self._vertex_normals_update:
            self._vertex_normals = vertex_normals(self.vertices, self.faces)
            self._vertex_normals_update = False
        return self._vertex_normals

    @property
    def face_textures(self):
        if self.texture_type in ['surface']:
            return self.textures
        elif self.texture_type in ['vertex']:
            return self.textures#srf.face_vertices(self.textures, self.faces)
        else:
            raise ValueError('texture type not applicable')

    def fill_back_(self):
        if not self._fill_back:
            self.faces = torch.cat((self.faces, self.faces[:, :, [2, 1, 0]]), dim=1)
            self.textures = torch.cat((self.textures, self.textures), dim=1)
            self._fill_back = True

    def reset_(self):
        self.vertices = self._origin_vertices
        self.faces = self._origin_faces
        self.textures = self._origin_textures
        self._fill_back = False
    
    @classmethod
    def from_obj(cls, filename_obj, normalization=False, load_texture=False, texture_res=1, texture_type='surface'):
        '''
        Create a Mesh object from a .obj file
        '''
        if load_texture:
            vertices, faces, textures, uvcoords, uvfaces = load_obj(filename_obj,
                                                     normalization=normalization,
                                                     texture_res=texture_res,
                                                     load_texture=True,
                                                     texture_type=texture_type)
        else:
            vertices, faces = load_obj(filename_obj,
                                           normalization=normalization,
                                           texture_res=texture_res,
                                           load_texture=False)
            textures = None
            uvcoords = None
            uvfaces = None
        return cls(vertices, faces, textures, uvcoords, uvfaces, texture_res, texture_type)

    def save_obj(self, filename_obj, save_texture=False, texture_res_out=16):
        if self.batch_size != 1:
            raise ValueError('Could not save when batch size >= 1')
        if save_texture:
            save_obj(filename_obj, self.vertices[0], self.faces[0], 
                         textures=self.textures[0],
                         texture_res=texture_res_out, texture_type=self.texture_type)
        else:
            save_obj(filename_obj, self.vertices[0], self.faces[0], textures=None)

    # def voxelize(self, voxel_size=32):
    #     face_vertices_norm = self.face_vertices * voxel_size / (voxel_size - 1) + 0.5
    #     return srf.voxelization(face_vertices_norm, voxel_size, False)

def write_obj_with_colors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        triangles: shape = (ntri, 3)
        colors: shape = (nver, 3)
    '''
    triangles = triangles.copy()
    triangles += 1 # meshlab start with 1
    
    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'
        
    # write obj
    with open(obj_name, 'w') as f:
        
        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 2], triangles[i, 1])
            f.write(s)

