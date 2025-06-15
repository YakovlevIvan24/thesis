import numpy as np
from collections import defaultdict

class Element:
    """Triangle finite element"""

    def __init__(self, node_ind: (int, int, int),
                 D: np.ndarray,
                rho: np.float64):
        self.node_ind = node_ind  # tuple of 4 node indices
        self.rho = rho  # density
        self.b = lambda t: np.zeros(3, dtype=np.float64)
        self.nu = 0
        self.E = 0
        self.H = np.zeros([12, 12])
        self.M = np.zeros([12,12])
        self.D = D  # tensor of elastic constants (6x6) Мне проще думать что он 3x3x3x3
        self.DB = np.zeros([6, 9], np.float64)  # matrix connecting stress and nodal displacements; tbc
        self.V = 0.0  # doubled element area; tbc
        self.sigma = np.zeros([6])  # stress tensor


    def to_string(self):
        return f"{self.node_ind}\nE = {self.E:.3f}, nu = {self.nu:.3f}\nrho = {self.rho:.3f}"


class Grid:
    """2D membrane in 3D space model"""
    supported_solvers = ["factorized", "no_inverse", "inverse"]

    def __init__(self, n_nodes: int, node_tags):
        # возможно node_tags не нужен
        # print(f'Nodes = {n_nodes}')
        self.reduced_dof_map = {}
        self.n_nodes = n_nodes  # number of nodes
        self.node_tags = node_tags
        self.free_nodes = []
        self.Dirichlet_nodes = []
        self.a = np.zeros([3 * n_nodes])  # vector of nodal displacements
        self.a_t = np.zeros([3 * n_nodes])  # vector of nodal velocities
        self.a_tt = np.zeros([3 * n_nodes])  # vector of nodal accelerations
        self.x_0 = np.zeros([n_nodes])
        self.y_0 = np.zeros([n_nodes])  # initial positions
        self.z_0 = np.zeros([n_nodes])
        self.elements = []  # list of elements, will be filled later
        self.D = [] # elastic moduli tensor
        self.Q = np.zeros([3,1]) # вектор сил
        self.H = np.zeros([3 * n_nodes, 3 * n_nodes], dtype=np.complex128)  # global stiffness matrix
        self.M = np.zeros([
            3 * n_nodes, 3 * n_nodes], dtype=np.complex128)  # global mass matrix
        self.F = np.zeros([3 * n_nodes,1], dtype=np.complex128)  # load vector
        self.constrained_vertices = set()
        self.node_groups = {}
        self.element_groups = {}

    def add_elem(self, elem: Element):
        if (max(*elem.node_ind) >= self.n_nodes) or (min(*elem.node_ind) < 0):
            raise ValueError("Element's node indices out of range")
        # TODO: check D matrix somehow
        if elem.rho <= 0:
            raise ValueError("Wrong element params:\n" + elem.to_string())
        self.elements.append(elem)

    def get_closest_vertex_index(self, loc: np.ndarray):
        if loc.shape[0] != 3:
            raise ValueError(f"loc must be a 3-d numpy ndarray [x, y]. Got {loc}")
        x, y, z = loc[0], loc[1], loc[2]
        dist = (self.x_0 - x) ** 2 + (self.y_0 - y) ** 2 + (self.z_0 - z) ** 2
        return np.argmin(dist)

    def clamped_boundary_x(self,x):
        return np.isclose(x[0], 0)

    def get_Dirichlet_nodes(self):
        x = [self.x_0,self.y_0,self.z_0]
        mask = self.clamped_boundary_x(x)
        masked = np.ma.masked_array(self.node_tags, np.invert(mask))
        self.Dirichlet_nodes = np.ma.compressed(masked)
        # self.free_nodes = np.delete(self.node_tags,self.Dirichlet_nodes)
        return np.ma.compressed(masked)


    def set_V(self):
        """"Calculates doubled area of each element; this will be used in further computations"""
        for elem in self.elements:
            i, j, m, p = elem.node_ind
            delta = np.array([[1.0, self.x_0[i], self.y_0[i],self.z_0[i]],
                              [1.0, self.x_0[j], self.y_0[j],self.z_0[j]],
                              [1.0, self.x_0[m], self.y_0[m],self.z_0[m]],
                              [1.0, self.x_0[p], self.y_0[p],self.z_0[p]]])

            elem.V = 1/6*np.linalg.det(delta)
            assert elem.V > 0


    def get_element_h(self, elem: Element):
        node_ids = elem.node_ind
        coords = np.array([
            [1, self.x_0[i], self.y_0[i], self.z_0[i]] for i in node_ids
        ])

        inv_coords = np.linalg.inv(coords)

        grads = inv_coords[1:, :]  # rows: [b_i; c_i; d_i], cols = nodes

        B = np.zeros((6, 12))
        for i in range(4):
            # I,J,M,P = elem.node_ind
            # a_i1 = np.linalg.det(np.array([[self.x_0[J], self.y_0[J], self.z_0[J]], [self.x_0[M], self.y_0[M], self.z_0[M]], [self.x_0[P], self.y_0[P], self.z_0[P]]]))/elem.V/6
            # b_i = -np.linalg.det(
            #     np.array([[1, self.y_0[J], self.z_0[J]], [1, self.y_0[M], self.z_0[M]], [1, self.y_0[P], self.z_0[P]]]))/elem.V/6
            # c_i = -np.linalg.det(
            #     np.array([[self.x_0[J], 1, self.z_0[J]], [self.x_0[M], 1, self.z_0[M]], [self.x_0[P], 1, self.z_0[P]]]))/elem.V/6
            # d_i = -np.linalg.det(
            #     np.array([[self.x_0[J], self.y_0[J], 1], [self.x_0[M], self.y_0[M], 1], [self.x_0[P], self.y_0[P], 1]]))/elem.V/6
            b_i, c_i, d_i = grads[0, i], grads[1, i], grads[2, i]
            B_i = np.array([
                [b_i, 0, 0],
                [0, c_i, 0],
                [0, 0, d_i],
                [c_i, b_i, 0],
                [0, d_i, c_i],
                [d_i, 0, b_i]
            ])
            B[:, 3 * i:3 * (i + 1)] = B_i

        h_e = B.T @ elem.D @ B * elem.V /36   # or divide by 36 if you define D accordingly
        return h_e

    def get_element_m(self, elem: Element):
        m_e = np.zeros([12,12])
        K_0 = 1/120*np.array([[2,1,1,1],[1,2,1,1],[1,1,2,1],[1,1,1,2]])

        for i in range(4):
            for j in range(4):
                m_ij =elem.rho * elem.V * 6   * K_0[i,j]
                m_e[3*i:3*(i+1),3*j:3*(j+1)] = np.eye(3)*m_ij
        return m_e

    def get_element_F(self, elem: Element):
        F_e = np.zeros([12,1])
        for i in range(4):
            j = (i + 1) % 4
            m = (i + 2) % 4
            p = (i + 3 ) % 4
            I, J, M, P = elem.node_ind[i], elem.node_ind[j], elem.node_ind[m], elem.node_ind[p]  # indices in external massive
            # todo Здесь написан бред, но пока Q = 0 это ни на что не влияет
            a_i = np.linalg.det(np.array([[self.x_0[J], self.y_0[J], self.z_0[J]], [self.x_0[J], self.y_0[M],self.z_0[M]], [self.x_0[J], self.y_0[P], self.z_0[P]]]))
            F_e[i:i+3] = -1/6/elem.V*a_i*self.Q
        return F_e


    def assemble_H(self):
        """"Calculates global stiffness matrix for entire grid; used in both static and dynamic problems"""
        visited_edges = set()
        for elem in self.elements:
            # elem.D = self.D
            h_e = self.get_element_h(elem)
            # h_e =(np.tensordot(A_1,B_1, axes=2).transpose() + np.tensordot(A_2,B_2, axes=2).transpose())
            for i in range(4):
                for j in range(4):
                    I, J = elem.node_ind[i], elem.node_ind[j]
                    self.H[3 * I:3 * (I + 1), 3 * J:3 * (J + 1)] += h_e[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
        return

    def assemble_F(self):
        """"Calculates global stiffness matrix for entire grid; used in both static and dynamic problems"""
        for elem in self.elements:
            elem.D = self.D
            F_e = self.get_element_F(elem)
            for i in range(4):
                I = elem.node_ind[i]
                self.F[3 * I:3 * (I + 1)] += F_e[3 * i:3 * (i + 1)]



    def assemble_M(self):
        visited_edges = set()
        for elem in self.elements:
            elem.D = self.D
            m_e = self.get_element_m(elem)
            for i in range(4):
                for j in range(4):
                    I, J = elem.node_ind[i], elem.node_ind[j]
                    self.M[3 * I:3 * (I + 1), 3 * J:3 * (J + 1)] += m_e[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
        return

    def compute_F(self, omega, u_D:dict):
        # print('diriclet called')
        # print(self.Dirichlet_nodes)
        visited_edges = set()
        F = self.F.copy()
        for elem in self.elements:
            elem.D = self.D
            h_e = self.get_element_h(elem)
            m_e = self.get_element_m(elem)
            for i in range(4):
                for j in range(4):
                    I, J = elem.node_ind[i], elem.node_ind[j]
                    if (I not in self.Dirichlet_nodes) and (J in self.Dirichlet_nodes):
                        # print(F[3 * I:3 * (I + 1)].shape)
                        F[3 * I:3 * (I + 1)] -= omega**2 * np.dot(m_e[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)], u_D[J])
                        F[3 * I:3 * (I + 1)] -= np.dot(h_e[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)], u_D[J])

        indices_to_delete = []
        for node in u_D.keys():
            base_index = 3 * node
            indices_to_delete.extend([base_index, base_index + 1, base_index + 2])

        mask = np.ones(3 * self.n_nodes, dtype=bool)
        mask[indices_to_delete] = False
        F = F[mask, :]
        return F

    def apply_Dirichlet_boundary(self, u_D: dict):


        indices_to_delete = []
        for node in u_D.keys():
            base_index = 3 * node
            indices_to_delete.extend([base_index, base_index + 1, base_index + 2])

        mask = np.ones(3 * self.n_nodes, dtype=bool)
        mask[indices_to_delete] = False

        # Save index map: from global DOF to reduced system DOF
        # self.reduced_dof_map = {}
        reduced_index = 0
        for global_index in range(3 * self.n_nodes):
            if mask[global_index]:
                self.reduced_dof_map[global_index] = reduced_index
                reduced_index += 1

        # Modify system
        self.H = self.H[mask, :][:, mask]
        self.M = self.M[mask, :][:, mask]
        # self.F = self.F[mask, :]
        # print("cond(H):", np.linalg.cond(self.H))
        # print("cond(M):", np.linalg.cond(self.M))
        # print("det(H):", np.linalg.det(self.H))
        # print("det(M):", np.linalg.det(self.M))
        return self.reduced_dof_map


    def ready(self):
        self.set_V()
        # self.set_DBmatrix()
        self.assemble_H()
        # self.assemble_f()
        self.assemble_M()
        self.assemble_F()
        # self.set_sigma()


def get_isotropic_elastic_tensor(E: np.float64, nu: np.float64) -> np.ndarray:


    coeff = E / ((1 + nu) * (1 - 2 * nu))

    M = np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
    ])

    C = coeff * M
    beta = 0.003
    assert np.isclose(C,C.transpose()).all()
    return C*(1+1j*beta)


def generate_from_points_and_triangles(node_tags,points:np.ndarray, tetrahedrons:np.ndarray,
        rho: np.float64, D=None, E=None, nu=None, node_groups=None, element_groups=None):

    if points.shape[1] != 3:
        points = points.transpose()
    if points.shape[1] != 3:
        # raise ArgumentException(f'Points must have shape (n_vertex, 2) or (2, n_vertex). Got {points.shape}')
        raise ValueError(f'Points must have shape (n_vertex, 2) or (2, n_vertex). Got {points.shape}')

    if D is None and (E is None or nu is None):
        raise ValueError("Either a full elastic tensor D or elastic constants (E, nu)"
                         "must be specified")
    if D is None:
        D = get_isotropic_elastic_tensor(E, nu)

    n_vertex = points.shape[0]
    res = Grid(n_vertex,node_tags)
    res.D = D.copy()

    res.x_0 = points[:, 0]
    res.y_0 = points[:, 1]
    res.z_0 = points[:, 2]

    for tetrahedron in tetrahedrons:
        tetrahedron = list(map(int, tetrahedron))

        node_ind = (tetrahedron[0], tetrahedron[1], tetrahedron[2], tetrahedron[3])
        el = Element(node_ind, D, rho)
        res.elements.append(el)

    if node_groups is not None:
        res.node_groups = node_groups

    if element_groups is not None:
        res.element_groups = element_groups

    return res

def generate_from_gmsh_mesh(mesh,
        rho, D=None, E=None, nu=None):
    points = mesh.points # only 2-d coords
    triangles = mesh.cells
    points_tags = mesh.points_tags
    return generate_from_points_and_triangles(points_tags,points, triangles, rho, D, E, nu)

