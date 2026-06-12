"""Convert TopologyGAN CSV dataset to OSFT 7-channel .npy format.

Original format (per sample, 171,841 values):
  [0*SN:1*SN]   VF (volume fraction, input)
  [1*SN:2*SN]   VM_stress (von Mises stress, input)
  [2*SN:3*SN]   load_x (input)
  [3*SN:4*SN]   load_y (input)
  [4*SN:4*SN+SE] output topology (target, element-wise, SE values)
  [4*SN+SE:...] FEM displacement/strain/stress (not needed for GAN)

OSFT 7-channel format (per sample, 7*SN values):
  Channel 0: VF (input)
  Channel 1: VM_stress (input)
  Channel 2: strain_energy (approximated from stress, input)
  Channel 3: output topology (target, node-wise)
  Channel 4: boundary conditions
  Channel 5: load_x
  Channel 6: load_y
"""

import numpy as np
import csv
import os
import argparse

H, W = 64, 128
SN = H * W
SE = (H - 1) * (W - 1)


def convert_csv_to_npy(csv_path: str, output_path: str, max_samples: int = None):
    """Convert a single CSV file to .npy format."""
    print(f"Reading {csv_path}...")

    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(np.array(row, dtype=np.float32))

    data = np.array(rows, dtype=np.float32)
    n_samples = len(data)
    print(f"  Samples: {n_samples}, Features: {data.shape[1]}")

    if max_samples:
        data = data[:max_samples]
        n_samples = max_samples

    # Allocate output
    output = np.zeros((n_samples, 7 * SN), dtype=np.float32)

    for i in range(n_samples):
        row = data[i]

        # Channel 0: VF (input)
        vf = row[0*SN:1*SN]
        output[i, 0*SN:1*SN] = vf

        # Channel 1: VM_stress (input)
        vm = row[1*SN:2*SN]
        output[i, 1*SN:2*SN] = vm

        # Channel 2: strain_energy — approximate from VM stress
        # Use VM_stress^2 / (2*E) as a rough proxy, normalize
        strain_energy = np.clip(vm * vm * 0.5, 0, 1)
        output[i, 2*SN:3*SN] = strain_energy

        # Channel 3: output topology (target)
        # Original is element-wise (SE values), expand to node-wise (SN)
        elem_output = row[4*SN:4*SN+SE]
        elem_2d = elem_output.reshape(H-1, W-1)
        # Pad to full node grid: output[node] = avg of connected elements
        node_output = np.zeros((H, W), dtype=np.float32)
        node_count = np.zeros((H, W), dtype=np.float32)
        for ii in range(H-1):
            for jj in range(W-1):
                val = elem_2d[ii, jj]
                node_output[ii, jj] += val
                node_output[ii+1, jj] += val
                node_output[ii, jj+1] += val
                node_output[ii+1, jj+1] += val
                node_count[ii, jj] += 1
                node_count[ii+1, jj] += 1
                node_count[ii, jj+1] += 1
                node_count[ii+1, jj+1] += 1
        node_output /= np.maximum(node_count, 1)
        output[i, 3*SN:4*SN] = node_output.ravel()

        # Channel 4: boundary conditions
        # Reconstruct from load_x/load_y: where loads are zero and at edges = BC
        lx = row[2*SN:3*SN]
        ly = row[3*SN:4*SN]
        bc = np.zeros(SN, dtype=np.float32)
        # Fixed nodes: where both loads are zero at the boundary
        lx_2d = lx.reshape(H, W)
        ly_2d = ly.reshape(H, W)
        bc_2d = np.zeros((H, W), dtype=np.float32)
        # Identify fixed edges: where lx and ly are both zero but adjacent to loaded areas
        # Simple heuristic: top/bottom rows with zero load = fixed
        for row_idx in range(H):
            for col_idx in range(W):
                if abs(lx_2d[row_idx, col_idx]) < 1e-6 and abs(ly_2d[row_idx, col_idx]) < 1e-6:
                    # Check if this node is at a potential boundary
                    if row_idx == 0:
                        bc_2d[row_idx, col_idx] = 1  # top edge fixed
                    elif row_idx == H - 1 and abs(vf.reshape(H, W)[row_idx, col_idx]) > 1e-6:
                        bc_2d[row_idx, col_idx] = 2  # bottom edge roller
        output[i, 4*SN:5*SN] = bc_2d.ravel()

        # Channel 5-6: loads
        output[i, 5*SN:6*SN] = lx
        output[i, 6*SN:7*SN] = ly

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples}...")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, output)
    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"Saved {n_samples} samples → {output_path} ({size_mb:.1f} MB)")

    # Print statistics
    print(f"\nStatistics:")
    for ch, name in enumerate(["VF", "VM_stress", "StrainE", "Output", "BC", "LoadX", "LoadY"]):
        ch_data = output[:, ch*SN:(ch+1)*SN]
        print(f"  {name}: mean={ch_data.mean():.4f}, std={ch_data.std():.4f}, "
              f"min={ch_data.min():.4f}, max={ch_data.max():.4f}")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/TopologyGAN/dataset/data_001.csv")
    parser.add_argument("--output", default="data/cantilever_train.npy")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    # Resolve path
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base, args.input)
    output_path = os.path.join(base, args.output)

    convert_csv_to_npy(input_path, output_path, args.max_samples)


if __name__ == "__main__":
    main()
