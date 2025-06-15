import matplotlib.pyplot as plt
import os
import gmsh
import numpy as np
import pandas as pd
from scipy import linalg as la
from scipy.sparse.linalg import cg, gmres, bicgstab, splu, bicg
from  final.plate3d import generate_from_gmsh_mesh

gmsh.initialize()
mesh_files = ['Mesh_202.msh']
# mesh_files = ['Mesh_132_t.msh','Mesh_192_t.msh','Mesh_202.msh','Mesh_228_t.msh','Mesh_325_t.msh','Mesh_414.msh','Mesh_513_t.msh', 'Mesh_708.msh', 'Mesh_845_t.msh', 'Mesh_1319.msh', 'Mesh_1762_t.msh', 'Mesh_2186.msh','Mesh_7682_t.msh']
for mesh_file in mesh_files:

    gmsh.open(mesh_file)
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
    grid = generate_from_gmsh_mesh(msh,7920,D=None,E=E,nu=nu)

    u_D = grid.get_Dirichlet_nodes()
    grid.ready()

    # Узлы дирихле - вроде верно работает, free egde - в коде учитывать не нужно.
    u_D = dict()
    for i in grid.get_Dirichlet_nodes():
        u_D[i] = np.array([0,0,1e0]).reshape((3,1))


    reduced_dof_map = grid.apply_Dirichlet_boundary(u_D)
    assert grid.H.shape[0] == len(reduced_dof_map)


    xtest, ytest, ztest = 110e-3, 10e-3, 1e-3
    test_ind = grid.get_closest_vertex_index(np.array([xtest,ytest,ztest]))
    # my_ind = test_ind - next((x[0] for x in enumerate(u_D.keys()) if x[1] > test_ind), len(u_D))
    global_dof_index = test_ind * 3 + 2  # z displacement of node

    if global_dof_index not in reduced_dof_map:
        print(f"Global DOF {global_dof_index} not in reduced system!")
    else:
        print(f"Mapped to reduced DOF: {reduced_dof_map[global_dof_index]}")
    reduced_dof_index = reduced_dof_map[global_dof_index]
    # print(my_ind *3 + 2)
    # print(reduced_dof_index)
    # Симметричные - только при реализации интегрироваеия через референсный элемент
    # assert np.allclose(grid.H[:,:], grid.H[:,:].T)
    # assert np.allclose(grid.M[:,:], grid.M[:,:].T)

    freq = np.linspace(0,2500,501)

    afc = np.zeros([len(freq)])
    # np.savetxt("K.csv", grid.H, delimiter=" ", fmt="%.5e")
    # np.savetxt("M.csv", grid.M, delimiter=" ", fmt="%.5e")

    # assert not np.any(np.all(grid.H == 0, axis=1))
    # assert not np.any(np.all(grid.H == 0, axis=0))
    # assert not np.any(np.all(grid.M == 0, axis=1))
    # assert not np.any(np.all(grid.M == 0, axis=0))

    # print(la.det(grid.H))

    eigvals, eigvecs = la.eigh(grid.H, grid.M)
    eigvals = np.sqrt(eigvals)
    frequencies = np.sqrt(eigvals) / (2 * np.pi)

    import time
    start = time.time()
    for i,f in enumerate(freq):
        print(f)
        omega = f * 2 * np.pi
        A = grid.H - omega**2 * grid.M
        F = grid.compute_F(omega, u_D)
        u = la.solve(A, F)
        afc[i] = np.abs(u[reduced_dof_index])

    end = time.time()
    elapsed = end - start  # elapsed time in seconds
    print(f"Elapsed time: {elapsed:.6f} seconds")
    # print(list(afc).index(max(afc)))
    plt.plot(freq,afc)
    plt.yscale('log')
    plt.title('Амплитудно-частотная характеристика')
    # plt.legend()
    plt.ylabel(r'$u_z$')
    plt.xlabel(r'f, Гц')
    afc_name = f'afc_{mesh_file[5:-4]}.png'
    plt.savefig(f'afc_test.png')
    plt.show()
    print(f'AFC saved as {afc_name}')
    # afc_data = 'afc_data.xlsx'
    # df = pd.read_excel(afc_data)
    # df[mesh_file] = afc
    # df.to_excel(afc_data,index=False)

