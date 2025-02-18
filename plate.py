import matplotlib.pyplot as plt
import numpy as np
from basix.ufl import element
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx.fem import Function, assemble_scalar, form, functionspace
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, apply_lifting, set_bc, create_vector
from dolfinx.io import XDMFFile, gmshio, VTKFile
from dolfinx import default_scalar_type
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells, BoundingBoxTree, compute_closest_entity, compute_distance_gjk, create_midpoint_tree

from ufl import dx, grad, inner, FacetNormal, SpatialCoordinate, Circumradius

from dolfinx import mesh, fem, plot, geometry


# Нужно для использования комплексных числел
print(f"Using {PETSc.ScalarType}.")



# 3d
gdim = 3

msh, ct, ft = gmshio.read_from_msh("new_mesh1.msh",MPI.COMM_WORLD,gdim=gdim)
# boundary_facets = mesh.exterior_facet_indices(msh.topology)
# 2d
print(f'ct = {ct}, ft = {ft}')
fdim = msh.topology.dim - 1

bf = "P3"
V = functionspace(msh,("P",3,(1, ))) # одномерное пространсво полиномов второй степени


def clamped_boundary(x):
    return np.isclose(x[0],0)



# boundary_facets = mesh.locate_entities_boundary(msh,fdim,clamped_boundary)
# print('boundary facets = ',boundary_facets)

u_D = np.array([1], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, ft.find(1)), V) # то же самое что строчкой ниже, в msh файле для границы задан physical group = 1
# bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)


h = 1*1e-3
E = 198*1e9
G = 77*1e9
# nu = E/2/G - 1
# D0 = fem.Constant(msh,default_scalar_type((E*(h**3)/12/(1-nu**2))))
beta = fem.Constant(msh,default_scalar_type(0.003))
rho = fem.Constant(msh,default_scalar_type(7920))

e = fem.Constant(msh,default_scalar_type(1/2*h))
f = fem.Constant(msh,default_scalar_type(0))
t = fem.Constant(msh, default_scalar_type(0))
# nu = fem.Constant(msh,default_scalar_type(nu))

nu = E/(2. * G) - 1
D0 = E*(h**3)/(12.*(1.-nu**2))


D11 = D0
D22 = D0
D66 = D0*(1-nu)/2
D12 = D0*nu
D16 = 0
D26 = 0
D_3d_real = ufl.as_matrix([[[[D11,1/2*D16,0],[D16,1/2*D26,0],[0,0,0]],[[1/2*D16,D12,0],[1/2*D26,D66,0],[0,0,0]],
                        [[0,0,0],[0,0,0],[0,0,0]]],[[[D16,1/2*D26,0],[D12,1/2*D26,0],[0,0,0]],
                        [[1/2*D26,D66,0],[1/2*D26,D22,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],
                        [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]])

D_2d_real = (ufl.as_matrix
    (
    [
    [[[D11,1/2*D16],[D16,1/2*D66]],
    [[1/2*D16,D12],[1/2*D66,D26]]],
    [[[D16,1/2*D66],[D12,1/2*D26]],
    [[1/2*D66,D26],[1/2*D26,D22]]]
    ]
    ))

# D_2d = D_2d_real*(1 + 1j*beta)
# D_2d = D_2d_real

D = D_3d_real
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

freqs = np.linspace(0,1000,1001,endpoint=True)
u_1 = np.zeros([len(freqs)])
u_la = np.zeros([len(freqs)])
u_2 = np.zeros([len(freqs)])


# print(grad(grad(u)[0,:]).ufl_shape)

for i in range(len(freqs)):
    Omega = freqs[i] * 2 * np.pi
    print(f'f = {freqs[i]}')
    omega = fem.Constant(msh,default_scalar_type(Omega))
    # Билинейная форма
    a = 2*e*(-rho * omega ** 2 * ( (ufl.inner(u, v) ) \
                + 1 / 3 * e ** 2 * ufl.inner(grad(u)[0,:], grad(v)[0,:]) ) \
              - 1/(2*e)*ufl.inner(D,ufl.outer(grad(grad(u)[0,:]),grad(grad(v)[0,:]))))*ufl.dx
    # Линейная форма
    L = ufl.inner(f,v[0])*ufl.dx

    # 1-й способ
    a1 = fem.form(a)
    L1 = fem.form(L)
    A = assemble_matrix(a1,bcs=[bc])
    A.assemble()
    b = create_vector(L1)

    assemble_vector(b,L1)
    set_bc(b,[bc])
    apply_lifting(b,[a1],bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # 2-й способ
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    sol = fem.Function(V, dtype=np.float64)

    solver.solve(b, sol.x.petsc_vec)
    sol.x.scatter_forward()


    problem = fem.petsc.LinearProblem(a,L,bcs=[bc])
    uh = problem.solve()
    xtest, ytest, ztest = 1e-2, 1e-2, 0
    # Ищется точка ближайшая к тестовой
    tree = bb_tree(msh, msh.geometry.dim)
    points = np.array([[xtest,ytest,ztest]])
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(msh, cell_candidates, points)
    front_cells = colliding_cells.links(0)
    print(f'test point index = {front_cells[:1]}')
    u_test_point = None
    u_test_point_2 = None
    if len(front_cells) > 0:
        print(front_cells[:1])
        u_test_point = uh.eval(points[0], front_cells[:1])
        u_test_point_2 = sol.eval(points[0], front_cells[:1])

    # print(f'solution = {np.absolute(uh.x.array[22])}')
    if u_test_point[0] in uh.x.array: # не очень понятно, почему в векторе нет значения, которое я вроде как достаю из этого вектора
        print('SUCCESS')


    u_1[i] = np.absolute(u_test_point[0]*1)
    u_2[i] = np.absolute(u_test_point_2[0] * 1)
    # _u[i] = np.abs(np.max(uh.x.array))
    print(f'u_1 =  {u_1[i]}')
    print(f'u_2 =  {u_2[i]}')

    # 3-й способ
    import numpy.linalg as la
    import numpy as np
    import jax.numpy as jnp

    # assert np.allclose(A[:,:], A[:,:].T, rtol=1e-5, atol=1e-8)

    u0 = jnp.linalg.solve(A[:, :], b[:])
    # assert jnp.all(np.linalg.eigvals(A[:,:]) > 0)
    # u0 = la.solve(A[:, :], b[:])
    u_1[i] = np.max(np.absolute(u0))
    # индекс искомого узла в векторе в первом случае - 22, => мб прокатит
    print(f'u_la =  {(np.absolute(u0[22]))}')
    u_la[i] = np.absolute(u0[22])

plt.plot(freqs,u_1,label='u_1')
plt.plot(freqs,u_2,label='u_2')
plt.plot(freqs,u_la,label='u_la')
# plt.yscale('log')
plt.xlabel('f, Hz')
plt.title('|u|')
plt.legend()
plt.ylim((0,2))
plt.savefig(f'afc_{bf}_D{gdim}.png')
plt.show()

# print(u_1)
# print(list(u_1).index(max(list(u_1))))

