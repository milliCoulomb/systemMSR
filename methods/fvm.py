# methods/fvm.py
import numpy as np
from scipy.sparse import diags

from .method import Method


class FVM(Method):
    """
    Finite volume discretisation of the 1D multigroup diffusion equations
    with transport of delayed neutron precursors and periodic boundary conditions.
    """

    name = f"FVM"

    def __init__(self, dxs):
        self.dx = dxs

    def build_stif(self, k):
        """
        Build the $-\div{D\grad\phi}$ operator:
        2 * D_i * [
            D_{i+1} / (D_i * \Delta x_{i+1} + D_{i+1} * \Delta x_i) * (\phi_i - \phi_{i+1}) -
            D_{i-1} / (D_i * \Delta x_{i-1} + D_{i-1} * \Delta x_i) * (\phi_{i-1} - \phi_i)
            ] / \Delta x_i
        """
        dxs = self.dx
        dxs_p = np.roll(dxs, 1)
        dxs_m = np.roll(dxs, -1)
        k_p = np.roll(k, 1)
        k_m = np.roll(k, -1)
        diag = 2 * k * (k_p / (k * dxs_p + k_p * dxs) + k_m / (k_m * dxs + k * dxs_m)) / dxs
        diag_p = -2 * k * k_p / (k * dxs_p + k_p * dxs) / dxs
        diag_m = -2 * k * k_m / (k_m * dxs + k * dxs_m) / dxs
        diagonals = [diag, diag_p[1:], diag_m[:-1], diag_p[0], diag_m[-1]]
        offsets = [0, 1, -1, 1 - len(dxs), len(dxs) - 1]
        return diags(diagonals, offsets, shape=(len(dxs), len(dxs)))

    def build_mass(self, m):
        """
        Build the $m(x)\phi(x)$ operator:
        m_i * \phi_i
        """
        dxs = self.dx
        return diags(m * np.ones_like(dxs), 0, shape=(len(dxs), len(dxs)))

    def build_grad(self, u):
        """
        Build the $v(x)\cdot\grad c(x)$ operator:
        v_i * (c_{i+1} - c_i) / \Delta x_i
        """
        dxs = self.dx
        u_m = np.roll(u, -1)
        diag = u / dxs
        diag_p = np.zeros_like(dxs)
        diag_m = -u_m / dxs
        diagonals = [diag, diag_p[1:], diag_m[:-1], diag_p[0], diag_m[-1]]
        offsets = [0, 1, -1, 1 - len(dxs), len(dxs) - 1]
        return diags(diagonals, offsets, shape=(len(dxs), len(dxs)))
