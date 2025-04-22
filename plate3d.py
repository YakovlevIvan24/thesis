from typing import List

import numpy as np
from matplotlib.patches import Polygon

# from scipy.spatial import Delaunay
from scipy import sparse as sp
import scipy.sparse.linalg
import scipy.linalg as lg

import sympy


import pickle

import pyvtk as pvtk

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
        self.D = D  # tensor of elastic constants (6x6) Мне проще думать что он 9x9
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
        self.H = np.zeros([3 * n_nodes, 3 * n_nodes], dtype=np.float64)  # global stiffness matrix
        self.M = np.zeros([
            3 * n_nodes, 3 * n_nodes], dtype=np.float64)  # global mass matrix
        self.F = np.zeros([3 * n_nodes,1])  # load vector
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
            raise ValueError(f"loc must be a 2-d numpy ndarray [x, y]. Got {loc}")
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
            elem.V = np.linalg.det(delta)
            if elem.V < 0.0:
                elem.node_ind = elem.node_ind[::-1]  # if for some reason they are in clockwise order instead
                elem.V *= -1.0  # of counter-clockwise, change order and fix area
            if elem.V == 0:
                print(f'i,j,m,p = {elem.node_ind} \n delta = {delta}')
                raise ValueError('V == 0')

    def get_element_h(self, elem: Element):
        h_e = np.zeros([12,12])
        for i in range(4):
            j = (i + 1) % 4
            m = (i + 2) % 4
            p = (i + 3 ) % 4
            I, J, M, P = elem.node_ind[i], elem.node_ind[j], elem.node_ind[m], elem.node_ind[p]  # indices in external massive
            # if I in self.Dirichlet_nodes or J in self.Dirichlet_nodes:
            #     continue

            b_i = -lg.det(np.array([[1, self.y_0[J], self.z_0[J]], [1, self.y_0[M],self.z_0[M]], [1, self.y_0[P], self.z_0[P]]]))
            c_i = -lg.det(np.array([[ self.x_0[J],1, self.z_0[J]], [self.x_0[M],1,self.z_0[M]], [self.x_0[P],1, self.z_0[P]]]))
            d_i = -lg.det(np.array([[self.x_0[J], self.y_0[J],1], [self.x_0[M],self.y_0[M],1], [self.x_0[P], self.y_0[P],1]]))

            assert not (b_i == 0 and c_i == 0 and d_i == 0)
            coef_i = [b_i, c_i, d_i]
            B_i = np.zeros([3,3,3])
            for l in range(3):
                B_i[:, :,l] = np.eye(3) * coef_i[l]

            coef_i = [b_i, c_i, d_i]
            B_i_132 = np.zeros([3,3,3])
            for l in range(3):
                B_i_132[:, l,:] = np.eye(3) * coef_i[l]

            b_j = -lg.det(np.array([[1, self.y_0[M], self.z_0[M]], [1, self.y_0[P],self.z_0[P]], [1, self.y_0[I], self.z_0[I]]]))
            c_j = -lg.det(np.array([[ self.x_0[M],1, self.z_0[M]], [self.x_0[P],1,self.z_0[P]], [self.x_0[I],1, self.z_0[I]]]))
            d_j = -lg.det(np.array([[self.x_0[M], self.y_0[M],1], [self.x_0[P],self.y_0[P],1], [self.x_0[I], self.y_0[I],1]]))

            assert not (b_j == 0 and c_j == 0 and d_j == 0)
            grad_N_T_j = np.zeros([3,3,3])
            coef_j = [b_j, c_j ,d_j]
            for l in range(3):
                grad_N_T_j[:, :,l] = np.eye(3) * coef_j[l]


            grad_N_T_132_j = np.zeros([3,3,3])
            for l in range(3):
                grad_N_T_132_j[:, l,:] = np.eye(3) * coef_j[l]

            A_j1 = np.tensordot(grad_N_T_132_j, elem.D, axes=2)
            A_j2 = np.tensordot(grad_N_T_j,elem.D, axes=2)

            assert not np.array_equal(B_i, np.zeros([3, 3, 3]))

            G1 = np.tensordot(A_j1,B_i,axes=2)
            G2 = np.tensordot(A_j2,B_i_132,axes=2)
            assert not np.array_equal(G1, np.zeros([3,3]))
            h_e[3*i:3*(i+1),3*j:3*(j+1)] = 1/2 / elem.V * (G1.transpose() + G2.transpose())
        return h_e

    def get_element_m(self, elem: Element):
        m_e = np.zeros([12,12])
        for i in range(4):

            j = (i + 1) % 4
            m = (i + 2) % 4
            p = (i + 3 ) % 4
            I, J, M, P = elem.node_ind[i], elem.node_ind[j], elem.node_ind[m], elem.node_ind[p]  # indices in external massive

            x = np.array([self.x_0[I], self.x_0[J], self.x_0[M], self.x_0[P]]) # ?
            y = np.array([self.y_0[I], self.y_0[J], self.y_0[M], self.y_0[P]])
            z = np.array([self.z_0[I], self.z_0[J], self.z_0[M], self.z_0[P]])
            x -= np.average(x)
            y -= np.average(y)  # switching to barycenter cords
            z -= np.average(z)

            a_i = lg.det(np.array([[self.x_0[J], self.y_0[J], self.z_0[J]], [self.y_0[M], self.y_0[M], self.z_0[M]], [self.z_0[P], self.y_0[P], self.z_0[P]]]))
            b_i = -lg.det(np.array([[1, self.y_0[J], self.z_0[J]], [1, self.y_0[M],self.z_0[M]], [1, self.y_0[P], self.z_0[P]]]))
            c_i = -lg.det(np.array([[ self.x_0[J],1, self.z_0[J]], [self.x_0[M],1,self.z_0[M]], [self.x_0[P],1, self.z_0[P]]]))
            d_i = -lg.det(np.array([[self.x_0[J], self.y_0[J],1], [self.x_0[M],self.y_0[M],1], [self.x_0[P], self.y_0[P],1]]))

            a_j = lg.det(np.array([[self.x_0[M], self.y_0[M], self.z_0[M]], [self.y_0[P], self.y_0[P], self.z_0[P]], [self.z_0[I], self.y_0[I], self.z_0[I]]]))
            b_j = -lg.det(np.array([[1, self.y_0[M], self.z_0[M]], [1, self.y_0[P],self.z_0[P]], [1, self.y_0[I], self.z_0[I]]]))
            c_j = -lg.det(np.array([[ self.x_0[M],1, self.z_0[M]], [self.x_0[P],1,self.z_0[P]], [self.x_0[I],1, self.z_0[I]]]))
            d_j = -lg.det(np.array([[self.x_0[M], self.y_0[M],1], [self.x_0[P],self.y_0[P],1], [self.x_0[I], self.y_0[I],1]]))

            def tetrahedral_bracket_sum(a, b):

                if len(a) != 4 or len(b) != 4:
                    raise ValueError("Input arrays must have 4 elements (tetrahedron vertices).")

                # Sum of a_i * b_i
                sum_sq = sum(a[k] * b[k] for k in range(4))

                # Sum of (a_i * b_j + a_j * b_i) for i < j
                sum_cross = 0
                for k in range(4):
                    for m in range(k + 1, 4):
                        sum_cross += (a[k] * b[m] + a[m] * b[k])

                return sum_sq + sum_cross

            coords_0 = [1, x[0], y[0], z[0]]
            coords = [x,y,z]
            coef_i = [a_i, b_i, c_i ,d_i]
            coef_j = [a_j, b_j, c_j, d_j]
            #todo переписать и проверить коэффициенты
            cross_sum = np.array([tetrahedral_bracket_sum(i, j) if not np.array_equal(i, j) else 0 for i in coords for j in coords])
            cross_coeff = np.array([i*j for i in coef_i[1:] for j in coef_j[1:]])
            cross_sum = np.sum(np.multiply(cross_sum,cross_coeff))
            direct_sum = np.sum([np.sum(i**2) + np.triu(i[:,None]-i).sum() for i in coords])
            zero_sum = np.sum([coef_i[i]*coords_0[i] for i in range(len(coef_i))]) + np.sum([coef_j[i]*coords_0[i] for i in range(len(coef_j))])
            m_ij = 1/36  * elem.rho / elem.V * (zero_sum + direct_sum + cross_sum)
            m_e[3*i:3*(i+1),3*j:3*(j+1)] = np.eye(3)*m_ij
        return m_e

    def get_element_F(self, elem: Element):
        F_e = np.zeros([12,1])
        for i in range(4):
            j = (i + 1) % 4
            m = (i + 2) % 4
            p = (i + 3 ) % 4
            I, J, M, P = elem.node_ind[i], elem.node_ind[j], elem.node_ind[m], elem.node_ind[p]  # indices in external massive

            a_i = lg.det(np.array([[self.x_0[J], self.y_0[J], self.z_0[J]], [self.x_0[J], self.y_0[M],self.z_0[M]], [self.x_0[J], self.y_0[P], self.z_0[P]]]))
            F_e[i:i+3] = -1/6/elem.V*a_i*self.Q
        return F_e


    def assemble_H(self):
        """"Calculates global stiffness matrix for entire grid; used in both static and dynamic problems"""
        for elem in self.elements:
            elem.D = self.D
            h_e = self.get_element_h(elem)
            for i in range(4):
                for j in range(4):
                    I, J = elem.node_ind[i], elem.node_ind[j]
                    if I in self.Dirichlet_nodes or J in self.Dirichlet_nodes:
                        continue

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
        for elem in self.elements:
            elem.D = self.D
            m_e = self.get_element_m(elem)
            for i in range(4):
                for j in range(4):
                    I, J = elem.node_ind[i], elem.node_ind[j]
                    if I in self.Dirichlet_nodes or J in self.Dirichlet_nodes:
                        continue
                    self.M[3 * I:3 * (I + 1), 3 * J:3 * (J + 1)] += m_e[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)]
        return


    def apply_Dirichlet_boundary(self,u_D: dict):
        # Так предлагается делать в Зинкевиче
        # for i in range(nodes.shape[0]):
        #     node = int(nodes[i])
        #     value = values[3*i:3*(i+1)]
        #     self.H[3*node:3*(node+1),3*node:3*(node+1)] *= 1e42
        #     self.M[3*node:3*(node+1),3*node:3*(node+1)] *= 1e42
        #     self.F[3*node:3*(node+1)] = 1e42*value

        # Другой подход
        for elem in self.elements:
            elem.D = self.D
            h_e = self.get_element_h(elem)
            m_e = self.get_element_m(elem)
            for i in range(4):
                for j in range(4):
                    I, J = elem.node_ind[i], elem.node_ind[j]
                    if not ((I not in self.Dirichlet_nodes) and (J in self.Dirichlet_nodes)):
                        continue
                    self.F[3 * I:3 * (I + 1)] -= np.dot(h_e[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)],u_D[J])
                    self.F[3 * I:3 * (I + 1)] -= np.dot(m_e[3 * i:3 * (i + 1), 3 * j:3 * (j + 1)],u_D[J])


        indices_to_delete = []
        for node in u_D.keys():
            base_index = 3 * node
            indices_to_delete.extend([base_index, base_index + 1, base_index + 2])


        mask = np.ones(3*self.n_nodes, dtype=bool)
        mask[indices_to_delete] = False

        # Плохо, но пока так
        self.H = self.H[mask, :][:, mask]
        self.M = self.M[mask, :][:, mask]
        self.F = self.F[mask, :]

        return


    def ready(self):
        self.set_V()
        # self.set_DBmatrix()
        self.assemble_H()
        # self.assemble_f()
        self.assemble_M()
        self.assemble_F()
        # self.set_sigma()


def get_isotropic_elastic_tensor(E: np.float64, nu: np.float64) -> np.ndarray:

    C0 = E / (1. + nu) / (1.0 - 2.0 * nu)
    C = C0 * np.array([ [[[1.-nu,nu,nu],[0.,0.,0.],[0.,0.,0.]],[[nu,1.-nu,nu],[0.,0.,0.],[0.,0.,0.]], #TODO ПРОВЕРИТЬ
                            [[nu,nu,1.-nu],[0.,0.,0.],[0.,0.,0.]]] ,[[[0.,0.,0.],[1.+nu,1.+nu,0.],[0.,0.,0.]],[[0.,0.,0.],[1.+nu,1.+nu,0.],[0.,0.,0.]],
                            [[0.,0.,0.],[0.,0.,1.+nu],[1.+nu,0.,0.]]],[[[0.,0.,0.],[0.,0.,1.+nu],[1.+nu,0.,0,]],[[0.,0.,0.],[0.,0.,0.],[0.,1.+nu,1.+nu]],[[0.,0.,0.],[0.,0.,0.],[0.,1.+nu,1.+nu]]],
                            ])
    return C


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

    res.x_0 += points[:, 0]
    res.y_0 += points[:, 1]
    res.z_0 += points[:, 2]

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

