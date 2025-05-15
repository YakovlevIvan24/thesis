import matplotlib.pyplot as plt

import plate3d
import os
import gmsh
import numpy as np
from scipy import linalg as la
from scipy.sparse.linalg import cg, gmres
from collections import Counter
gmsh.initialize()
gmsh.open('new_mesh_tet_only.msh')
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
node_tags -=1
node_tags = node_tags.astype(int)


# Костыль, чтоб меньше переписывать функцию создания сетки
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
grid = plate3d.generate_from_gmsh_mesh(msh,7920,D=None,E=E,nu=nu)

u_D = grid.get_Dirichlet_nodes()
grid.ready()

# Узлы дирихле - вроде верно работает, free egde - в коде учитывать не нужно.
# print(f'u_D = {u_D}')
u_D = dict()
for i in grid.get_Dirichlet_nodes():
    u_D[i] = np.array([0,0,1e0]).reshape((3,1))

grid.apply_Dirichlet_boundary(u_D)


for i in range(grid.M.shape[0]):
    for j in range(i+1,grid.M.shape[0]):
        if np.array_equal(grid.H[i,:],grid.H[j,:]):
            print(f'{i}{j}')

xtest, ytest, ztest = 11e-2, 1e-2, 1e-3
test_ind = grid.get_closest_vertex_index(np.array([xtest,ytest,ztest]))
test_ind -= next((x[0] for x in enumerate(u_D.keys()) if x[1] > test_ind), len(u_D))


# Симметричные - только при реализации интегрироваеия через референсный элемент
assert np.allclose(grid.H[:,:], grid.H[:,:].T)
assert np.allclose(grid.M[:,:], grid.M[:,:].T)


f = np.linspace(100,900,int(801*2*np.pi))
# Omega = f * 2*np.pi
Omega = np.linspace(0,1000,1001)
afc = np.zeros([len(Omega)])
for i in range(len(Omega)):

    A = grid.H - Omega[i]**2 * grid.M
    u = la.solve(A, grid.F)
    # u, _ = gmres(A, grid.F, rtol=1e-5)
    # Это я написал в творческом порыве, перемещение - вертор 3x1 -> ind *=3, интересует третья компонента -> ind +=2
    afc[i] = np.abs(u[test_ind*3+2])


plt.plot(Omega,afc)
plt.yscale('log')

plt.title('Амплитудно-частотная характеристика')
plt.ylabel(r'$u_z$, mm')
plt.xlabel(r'$\omega$,  $s^{-1}$')
plt.savefig('afc.png')
plt.show()
