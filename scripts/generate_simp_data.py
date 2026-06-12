"""Generate multi-structure topology optimization data via SIMP.

Full-resolution 88-line SIMP (64×128), parallelized via multiprocessing.

Usage:
  python scripts/generate_simp_data.py --problems "MBB Beam,L-Beam,Bridge" --n-samples 300 --workers 8
"""

import numpy as np
import os, sys, time, argparse
from multiprocessing import Pool

from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import minres

# ── constants ──────────────────────────────────────────────────
H_IMG, W_IMG = 64, 128          # output image size
SN = H_IMG * W_IMG              # 8192
E0, Emin, nu = 1.0, 1e-9, 0.3  # material

# unit stiffness (plane stress, E=1.0)
KE = np.array([
    [1/2-nu/6,  1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8, -1/4+nu/12, -1/8-nu/8,  nu/6,       1/8-3*nu/8],
    [1/8+nu/8,  1/2-nu/6,  1/8-3*nu/8,  nu/6,       -1/8-nu/8,  -1/4+nu/12, -1/8+3*nu/8, -1/4-nu/12],
    [-1/4-nu/12,1/8-3*nu/8, 1/2-nu/6,  -1/8-nu/8,    nu/6,       1/8+nu/8,  -1/4+nu/12,  1/8-3*nu/8],
    [-1/8+3*nu/8,nu/6,     -1/8-nu/8,   1/2-nu/6,    1/8-3*nu/8, -1/4+nu/12,  1/8+nu/8,  -1/4-nu/12],
    [-1/4+nu/12,-1/8-nu/8,  nu/6,       1/8-3*nu/8,  1/2-nu/6,   1/8+nu/8,  -1/4-nu/12, -1/8+3*nu/8],
    [-1/8-nu/8, -1/4+nu/12, 1/8+nu/8,  -1/4+nu/12,   1/8+nu/8,   1/2-nu/6,  -1/8+3*nu/8,  nu/6],
    [nu/6,     -1/8+3*nu/8,-1/4+nu/12,  1/8+nu/8,   -1/4-nu/12, -1/8+3*nu/8,  1/2-nu/6,  -1/8-nu/8],
    [1/8-3*nu/8,-1/4-nu/12,1/8-3*nu/8, -1/4-nu/12,  -1/8+3*nu/8,  nu/6,      -1/8-nu/8,   1/2-nu/6],
], dtype=np.float64) * (E0 / (1 - nu**2))


# ── SIMP solver ─────────────────────────────────────────────────
def simp_2d(nelx, nely, volfrac, penalty, rmin, bc, max_iter=100):
    """Full-resolution SIMP for 2D topology optimization.

    Args:
        nelx, nely: element counts (128, 64)
        bc: dict with 'fixed_dofs', 'loaded_dofs', 'forces'
    Returns:
        x: density [nely, nelx], compliance: float
    """
    nn = (nelx + 1) * (nely + 1)   # total nodes
    ndof = 2 * nn

    # node numbering: nodenrs[y, x] = y * (nelx+1) + x
    npr = nelx + 1
    nodenrs = np.arange(nn).reshape(nely + 1, nelx + 1)

    # edofMat: [nel, 8]
    edof = np.zeros((nelx * nely, 8), dtype=np.int32)
    n = 0
    for ex in range(nelx):
        for ey in range(nely):
            n1 = nodenrs[ey, ex]
            n2 = nodenrs[ey + 1, ex]
            n3 = nodenrs[ey + 1, ex + 1]
            n4 = nodenrs[ey, ex + 1]
            edof[n] = [2*n1,2*n1+1, 2*n2,2*n2+1, 2*n3,2*n3+1, 2*n4,2*n4+1]
            n += 1

    # filter
    r_int = int(np.ceil(rmin))
    iH, jH, sH = [], [], []
    for i in range(nelx):
        for j in range(nely):
            e1 = j * nelx + i
            for k in range(max(i - r_int, 0), min(i + r_int + 1, nelx)):
                for m in range(max(j - r_int, 0), min(j + r_int + 1, nely)):
                    e2 = m * nelx + k
                    fac = rmin - np.sqrt((i - k)**2 + (j - m)**2)
                    if fac > 0:
                        iH.append(e1); jH.append(e2); sH.append(fac)
    Hf = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsr()
    Hs = np.array(Hf.sum(axis=1)).flatten()

    # BC
    fixed = bc['fixed_dofs']
    free = np.setdiff1d(np.arange(ndof), fixed)
    F = np.zeros(ndof)
    for dof, val in zip(bc['loaded_dofs'], bc['forces']):
        F[dof] = val

    # COO prealloc
    nel = nelx * nely
    rows = np.zeros(nel * 64, dtype=np.int32)
    cols = np.zeros(nel * 64, dtype=np.int32)
    vals = np.zeros(nel * 64, dtype=np.float64)

    x = np.ones((nely, nelx)) * volfrac + np.random.rand(nely, nelx) * 0.01
    U = np.zeros(ndof)
    change = 1.0
    loop = 0

    while change > 0.005 and loop < max_iter:
        loop += 1

        # ── FE assembly ──
        nz = 0
        for ey in range(nely):
            for ex in range(nelx):
                idx = ey * nelx + ex
                ed = edof[idx]
                Ee = Emin + x[ey, ex] ** penalty * (E0 - Emin)
                factor = Ee / E0
                for ii in range(8):
                    for jj in range(8):
                        rows[nz] = ed[ii]
                        cols[nz] = ed[jj]
                        vals[nz] = KE[ii, jj] * factor
                        nz += 1

        K = coo_matrix((vals[:nz], (rows[:nz], cols[:nz])), shape=(ndof, ndof)).tocsr()
        Kff = K[free][:, free] + eye(len(free)) * Emin * 10
        try:
            U[free], _ = minres(Kff, F[free], rtol=1e-8, maxiter=5000)
        except Exception:
            U.fill(0)

        # ── compliance + sensitivity ──
        c = 0.0
        dc = np.zeros((nely, nelx))
        for ey in range(nely):
            for ex in range(nelx):
                ed = edof[ey * nelx + ex]
                Ue = U[ed]
                xp = x[ey, ex]
                factor = (Emin + xp ** penalty * (E0 - Emin)) / E0
                ce = 0.0
                for ii in range(8):
                    for jj in range(8):
                        ce += Ue[ii] * KE[ii, jj] * Ue[jj]
                c += factor * ce
                dc[ey, ex] = -penalty * xp ** (penalty - 1) * (E0 - Emin) / E0 * ce

        # ── filter ──
        dc = (Hf @ dc.ravel()).reshape(nely, nelx) / (Hs.reshape(nely, nelx) + 1e-12)

        # ── OC update ──
        l1, l2 = 0.0, 1e9
        move = 0.2
        target = volfrac * nelx * nely
        while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-6:
            lm = 0.5 * (l1 + l2)
            xn = np.maximum(0.0, np.maximum(x - move,
                  np.minimum(1.0, np.minimum(x + move,
                  x * np.sqrt(np.maximum(-dc, 0) / (lm + 1e-12))))))
            if np.sum(xn) > target:
                l1 = lm
            else:
                l2 = lm

        change = np.max(np.abs(xn - x))
        x = xn

    return x, float(c)


# ── BC configs ──────────────────────────────────────────────────
def get_bc_config(problem, volfrac, nelx=128, nely=64):
    """Return {'fixed_dofs', 'loaded_dofs', 'forces'}.

    Node index: nodenrs[y, x] = y * (nelx+1) + x
    """
    npr = nelx + 1

    if problem == "Cantilever":
        # Left edge clamped, load right edge center (downward)
        fixed = []
        for y in range(nely + 1):
            nid = y * npr        # col=0
            fixed.extend([2 * nid, 2 * nid + 1])
        cy = nely // 2
        lx = nelx
        loaded = [2 * (cy * npr + lx) + 1]
        force = [-1.0]

    elif problem == "MBB Beam":
        # right edge clamped, load mid-left
        fixed = []
        for y in range(nely + 1):
            nid = y * npr + nelx
            fixed.extend([2 * nid, 2 * nid + 1])
        ly = np.clip(nely // 2 + np.random.randint(-nely // 8, nely // 8 + 1), 1, nely - 1)
        loaded = [2 * (ly * npr) + 1]
        force = [-1.0]

    elif problem == "L-Beam":
        # top edge clamped, load mid-right
        fixed = []
        for x in range(nelx + 1):
            nid = nely * npr + x
            fixed.extend([2 * nid, 2 * nid + 1])
        loaded = [2 * ((nely // 2) * npr + nelx) + 1]
        force = [-1.0]

    elif problem == "Bridge":
        # bottom edge clamped, distributed top load
        fixed = []
        for x in range(nelx + 1):
            fixed.extend([2 * x, 2 * x + 1])
        loaded, force = [], []
        for x in range(nelx // 4, 3 * nelx // 4 + 1, 3):
            loaded.append(2 * (nely * npr + x) + 1)
            force.append(-0.15)
    else:
        raise ValueError(f"Unknown: {problem}")

    return {
        'fixed_dofs': np.array(fixed, dtype=np.int32),
        'loaded_dofs': np.array(loaded, dtype=np.int32),
        'forces': np.array(force, dtype=np.float64),
    }


# ── sample generator (importable, for multiprocessing) ──────────
def generate_one(args):
    """Generate one 7-channel topology sample. Called by Pool workers."""
    problem, seed = args
    np.random.seed(seed)
    volfrac = np.random.uniform(0.3, 0.6)

    # SIMP at full resolution
    bc = get_bc_config(problem, volfrac, nelx=128, nely=64)
    density, _ = simp_2d(128, 64, volfrac, 3.0, 1.5, bc, max_iter=50)

    # element density → node density
    nd = np.zeros((64, 128), dtype=np.float32)
    nc = np.zeros((64, 128), dtype=np.float32)
    for i in range(63):
        for j in range(127):
            v = density[i, j]
            nd[i, j] += v; nd[i+1, j] += v; nd[i, j+1] += v; nd[i+1, j+1] += v
            nc[i, j] += 1; nc[i+1, j] += 1; nc[i, j+1] += 1; nc[i+1, j+1] += 1
    nd /= np.maximum(nc, 1)

    # 7-channel layout
    vf  = np.full(SN, volfrac, dtype=np.float32)
    vm  = (1.0 - nd.ravel()) * np.random.uniform(0.8, 1.2, SN).astype(np.float32)
    se  = np.clip(0.5 * vm * vm * 0.5, 0, 1).astype(np.float32)

    bc_img  = np.zeros(SN, dtype=np.float32)
    lx_img  = np.zeros(SN, dtype=np.float32)
    ly_img  = np.zeros(SN, dtype=np.float32)

    # BC → image pixels. FEM mesh has 129×65 nodes but image is 128×64.
    # Boundary nodes at x=128 or y=64 are clamped to image edge.
    npr_fem = 129
    for dof in bc['fixed_dofs']:
        nid, d = dof // 2, dof % 2
        y_f, x_f = nid // npr_fem, nid % npr_fem
        yi, xi = min(y_f, 63), min(x_f, 127)
        px = yi * 128 + xi
        if d == 0:   bc_img[px] += 1   # bit 0: X-fixed
        else:        bc_img[px] += 2   # bit 1: Y-fixed
    for dof, f in zip(bc['loaded_dofs'], bc['forces']):
        nid, d = dof // 2, dof % 2
        y_f, x_f = nid // npr_fem, nid % npr_fem
        yi, xi = min(y_f, 63), min(x_f, 127)
        px = yi * 128 + xi
        if d == 0:   lx_img[px] = f
        else:        ly_img[px] = f

    sample = np.zeros(7 * SN, dtype=np.float32)
    sample[0*SN:1*SN] = vf
    sample[1*SN:2*SN] = vm
    sample[2*SN:3*SN] = se
    sample[3*SN:4*SN] = nd.ravel()
    sample[4*SN:5*SN] = bc_img
    sample[5*SN:6*SN] = lx_img
    sample[6*SN:7*SN] = ly_img
    return sample


# ── main ────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--problems", default="MBB Beam,L-Beam,Bridge")
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output-dir", default="data")
    args = p.parse_args()

    problems = [s.strip() for s in args.problems.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    t_total = time.time()

    for prob in problems:
        fname = os.path.join(args.output_dir,
                             f"{prob.lower().replace(' ', '_')}_train.npy")
        print(f"\n{prob}: {args.n_samples} samples, {args.workers} workers → {fname}")
        t0 = time.time()

        tasks = [(prob, i) for i in range(args.n_samples)]
        with Pool(args.workers) as pool:
            results = pool.map(generate_one, tasks)

        data = np.array(results, dtype=np.float32)
        np.save(fname, data)
        mb = os.path.getsize(fname) / 1024**2
        elapsed = (time.time() - t0) / 60
        print(f"  done: {mb:.0f} MB in {elapsed:.1f} min "
              f"({elapsed * 60 / args.n_samples:.0f} s/sample)")

    total = (time.time() - t_total) / 60
    print(f"\nAll done: {total:.1f} min")


if __name__ == "__main__":
    main()
