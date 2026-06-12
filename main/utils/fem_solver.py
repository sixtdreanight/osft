"""FEM compliance solver for topology optimization validation.

Uses Solidspy 1.1 for structural analysis. Treats the topology image
(H×W pixels) as ELEMENT densities with (H+1)×(W+1) FEM nodes.
"""

import numpy as np
from typing import Dict


class FEMSolver:
    """Compute structural compliance for topology density fields.

    Mesh: (H+1)×(W+1) nodes, H×W quad elements.
    Topology image [H, W] is interpreted as per-element density.
    """

    def __init__(self, height: int = 64, width: int = 128):
        self.H = height
        self.W = width
        self.n_nodes = (height + 1) * (width + 1)
        self.n_elem = height * width
        # Precompute element node connectivity
        self._elem_nodes = _make_elem_nodes(height, width)

    def compute_compliance(
        self,
        topology: np.ndarray,
        bc_x: np.ndarray,
        bc_y: np.ndarray,
        load_x: np.ndarray,
        load_y: np.ndarray,
    ) -> float:
        """Compute FEM compliance for a single topology.

        Args:
            topology: Element density field [H, W] or [H*W], values in [0, 1]
            bc_x: X-direction BC at NODES [(H+1)*(W+1)], -1=fixed, 0=free
            bc_y: Y-direction BC at NODES [(H+1)*(W+1)], -1=fixed, 0=free
            load_x: X-direction nodal loads [(H+1)*(W+1)]
            load_y: Y-direction nodal loads [(H+1)*(W+1)]

        Returns:
            Compliance value (scalar)
        """
        import solidspy.assemutil as ass
        import solidspy.solutil as sol
        import solidspy.postprocesor as pos

        topo = topology.reshape(self.H, self.W).ravel()
        bcx = bc_x.reshape(-1)
        bcy = bc_y.reshape(-1)
        lx = load_x.reshape(-1)
        ly = load_y.reshape(-1)

        # Constraints [n_nodes, 2]: bc_x, bc_y
        cons = np.column_stack([bcx, bcy])

        # Elements [n_elem, 7]: id, type=1(quad), mat_id, n0, n1, n2, n3
        elements = np.zeros([self.n_elem, 7], dtype=int)
        elements[:, 0] = range(self.n_elem)
        elements[:, 1] = 1  # quad element
        elements[:, 2] = range(self.n_elem)  # one material per element
        elements[:, 3:] = self._elem_nodes

        # SIMP: E = E_min + rho^p * (E0 - E_min)
        E_min, E0, p = 1e-9, 1.0, 3.0
        E_elem = E_min + (topo ** p) * (E0 - E_min)
        mats = np.column_stack([E_elem, np.full(self.n_elem, 0.3)])  # E, nu

        # Loads
        load_mask = (lx ** 2 + ly ** 2) > 1e-8
        load_nodes = np.where(load_mask)[0]
        if len(load_nodes) == 0:
            return 0.0
        loads = np.column_stack([load_nodes, lx[load_nodes], ly[load_nodes]])

        # Node coordinates: (H+1)×(W+1) grid
        nodes = np.zeros([self.n_nodes, 3], dtype=np.float64)
        nodes[:, 0] = np.arange(self.n_nodes, dtype=np.float64)
        nodes[:, 1] = _make_node_x(self.H, self.W)  # x coordinates
        nodes[:, 2] = _make_node_y(self.H, self.W)  # y coordinates

        # Assemble and solve
        DME, IBC, neq = ass.DME(cons, elements)
        KG, _ = ass.assembler(elements, mats, nodes, neq, DME)
        # Regularize near-singular stiffness from void elements (E ≈ 1e-9)
        import scipy.sparse as sp
        if sp.issparse(KG):
            KG += sp.eye(KG.shape[0], format='csr') * 1e-7
        else:
            KG += np.eye(KG.shape[0]) * 1e-7
        RHSG = ass.loadasem(loads, IBC, neq)
        UG = sol.static_sol(KG, RHSG)
        UC = pos.complete_disp(IBC, nodes, UG)

        # Compliance = sum(F_i * U_i)
        compliance = 0.0
        for row in loads:
            nid = int(row[0])
            compliance += row[1] * UC[nid, 0] + row[2] * UC[nid, 1]

        return float(abs(compliance))

    def compute_physics_metrics(
        self,
        fake_topology: np.ndarray,
        real_topology: np.ndarray,
        bc: np.ndarray,
        load_x: np.ndarray,
        load_y: np.ndarray,
    ) -> Dict[str, float]:
        """Compare physical fidelity of fake vs real topology.

        BC convention (raw, un-normalized values):
          1 = X-fixed, 2 = Y-fixed, 3 = X+Y fixed (1+2).
          bc/load_x/load_y are stored at IMAGE resolution [H, W].
          We upscale to FEM node resolution [(H+1), (W+1)].
        """
        # Upscale BC and loads from image [H,W] to FEM nodes [(H+1),(W+1)]
        bc_nodes = _upscale_to_nodes(bc.reshape(self.H, self.W))
        lx_nodes = _upscale_to_nodes(load_x.reshape(self.H, self.W))
        ly_nodes = _upscale_to_nodes(load_y.reshape(self.H, self.W))

        bc_x = np.where(np.isclose(bc_nodes, 1.0) | np.isclose(bc_nodes, 3.0),
                        -1.0, 0.0)
        bc_y = np.where(np.isclose(bc_nodes, 2.0) | np.isclose(bc_nodes, 3.0),
                        -1.0, 0.0)

        try:
            comp_fake = self.compute_compliance(
                fake_topology, bc_x, bc_y, lx_nodes, ly_nodes)
        except Exception:
            comp_fake = float("nan")

        try:
            comp_real = self.compute_compliance(
                real_topology, bc_x, bc_y, lx_nodes, ly_nodes)
        except Exception:
            comp_real = float("nan")

        vf_fake = float(np.mean(fake_topology))
        vf_real = float(np.mean(real_topology))
        vf_error = abs(vf_fake - vf_real) / (vf_real + 1e-8)
        comp_error = (
            abs(comp_fake - comp_real) / (comp_real + 1e-8)
            if comp_real > 1e-10
            else float("nan")
        )

        return {
            "compliance_fake": comp_fake,
            "compliance_real": comp_real,
            "compliance_error": comp_error,
            "vf_fake": vf_fake,
            "vf_real": vf_real,
            "vf_error": vf_error,
        }


def _make_elem_nodes(H: int, W: int) -> np.ndarray:
    """Build element connectivity for H×W quad mesh.

    Node numbering (row-major): node_id = row * (W+1) + col
    Returns [n_elem, 4]: [bottom-left, bottom-right, top-right, top-left]
    """
    elems = np.zeros([H * W, 4], dtype=int)
    for row in range(H):
        for col in range(W):
            idx = row * W + col
            nw = row * (W + 1) + col          # bottom-left
            ne = row * (W + 1) + col + 1      # bottom-right
            se = (row + 1) * (W + 1) + col + 1  # top-right
            sw = (row + 1) * (W + 1) + col      # top-left
            elems[idx] = [nw, ne, se, sw]
    return elems


def _make_node_x(H: int, W: int) -> np.ndarray:
    """X-coordinates of (H+1)×(W+1) nodes."""
    xs = np.zeros((H + 1) * (W + 1), dtype=np.float64)
    for row in range(H + 1):
        for col in range(W + 1):
            xs[row * (W + 1) + col] = float(col)
    return xs


def _make_node_y(H: int, W: int) -> np.ndarray:
    """Y-coordinates of (H+1)×(W+1) nodes."""
    ys = np.zeros((H + 1) * (W + 1), dtype=np.float64)
    for row in range(H + 1):
        for col in range(W + 1):
            ys[row * (W + 1) + col] = float(row)
    return ys


def _upscale_to_nodes(img: np.ndarray) -> np.ndarray:
    """Upscale image [H,W] to FEM nodes [(H+1),(W+1)].

    Image stores values at pixel centers. FEM nodes are at pixel corners.
    Interior nodes inherit from surrounding pixels; boundary nodes use
    nearest pixel value.
    """
    H, W = img.shape
    nodes = np.zeros((H + 1) * (W + 1), dtype=img.dtype)
    nodes_2d = nodes.reshape(H + 1, W + 1)

    # Interior nodes: average of 4 surrounding pixels
    nodes_2d[1:H, 1:W] = (
        img[:-1, :-1] + img[1:, :-1] + img[:-1, 1:] + img[1:, 1:]) / 4.0

    # Edge nodes (not corners): average of 2 adjacent pixels
    nodes_2d[0, 1:W] = (img[0, :-1] + img[0, 1:]) / 2.0       # top edge
    nodes_2d[-1, 1:W] = (img[-1, :-1] + img[-1, 1:]) / 2.0    # bottom edge
    nodes_2d[1:H, 0] = (img[:-1, 0] + img[1:, 0]) / 2.0       # left edge
    nodes_2d[1:H, -1] = (img[:-1, -1] + img[1:, -1]) / 2.0    # right edge

    # Corner nodes: nearest pixel
    nodes_2d[0, 0] = img[0, 0]
    nodes_2d[0, -1] = img[0, -1]
    nodes_2d[-1, 0] = img[-1, 0]
    nodes_2d[-1, -1] = img[-1, -1]

    return nodes
