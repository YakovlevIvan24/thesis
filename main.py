import matplotlib.pyplot as plt

import plate3d
from plate3d import generate_from_gmsh_mesh
import gmsh
import numpy as np
from scipy import linalg as la
from collections import Counter
gmsh.initialize()
gmsh.open('new_mesh_tet_only.msh')
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
node_tags -=1
node_tags = node_tags.astype(int)

msh = None
class Mesh:
    def __init__(self,cells,points,points_tags):
        self.cells = cells
        self.points = points
        self.points_tags = points_tags


points = np.reshape(np.split(node_coords,len(node_coords)//3), (len(node_coords)//3,3))


elementType = 4  # 4 = tetrahedron
elementTags, nodeTags = gmsh.model.mesh.getElementsByType(elementType)
nodeTags -=1
cells = np.array(nodeTags).reshape(-1, 4)
points_dict = dict(zip(node_tags,points))
points_dict = dict(sorted(points_dict.items()))
points = np.reshape(np.array(list(points_dict.values())),(len(points//3),3))

msh = Mesh(cells,points, node_tags)

E = 198*1e9
G = 77*1e9
nu = E/2/G-1
grid = generate_from_gmsh_mesh(msh,1e3,D=None,E=E,nu=nu)

grid.ready()

u_D = grid.get_Dirichlet_nodes()

grid.apply_Dirichlet_boundary(u_D, np.full([3*u_D.shape[0],1],1))

for i in range(grid.M.shape[0]):
    for j in range(i+1,grid.M.shape[0]):
        if np.array_equal(grid.M[i,:],grid.M[j,:]):
            print(f'row {i} = row {j}')

omega = 100

A = grid.H - omega**2 * grid.M
u = la.solve(A,grid.F)

print(u)

