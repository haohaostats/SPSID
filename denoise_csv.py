
#!/usr/bin/env python
# denoise_csv.py
# Minimal CLI: read an N×N weighted adjacency matrix (CSV/NPY),
# run SPSID, and write the denoised matrix as CSV.

import argparse
import os
import numpy as np
import pandas as pd

from methods import spsid


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Denoise a weighted adjacency matrix using SPSID.\n\n"
            "Example:\n"
            "  python denoise_csv.py --input data/example_W_obs.csv\n"
            "  python denoise_csv.py -i my_network.csv -o my_network_SPSID.csv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to input adjacency matrix (CSV or NPY)."
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path to output CSV (default: <input>_SPSID.csv in the same folder)."
    )
    parser.add_argument(
        "--lambda_val", type=float, default=1000.0,
        help="Shrinkage parameter λ (default: 1000)."
    )
    parser.add_argument(
        "--eps1", type=float, default=1e-6,
        help="Row-regularisation epsilon_1 (default: 1e-6)."
    )
    parser.add_argument(
        "--eps2", type=float, default=1e-6,
        help="Diagonal-regularisation epsilon_2 (default: 1e-6)."
    )
    return parser.parse_args()


def load_matrix(input_path: str):
    """
    加载 N×N 网络矩阵：
    - 如果是 CSV：第一列认为是 index，列名为节点名，需要是 N×N。
    - 如果是 NPY：直接读 numpy 数组，节点名默认 node_0, node_1, ...
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext in [".csv", ".tsv"]:
        df = pd.read_csv(input_path, index_col=0)
        if df.shape[0] != df.shape[1]:
            raise ValueError(
                f"Input matrix is not square: {df.shape}. "
                "denoise_csv.py expects an N×N adjacency matrix."
            )
        node_names = list(df.index)
        # 如果 index 和 columns 不一致，尝试对齐一下
        if list(df.columns) != node_names:
            df = df.loc[node_names, node_names]
        mat = df.values.astype(float)
        return mat, node_names
    elif ext == ".npy":
        mat = np.load(input_path).astype(float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"NPY array must be 2D and square, got shape {mat.shape}."
            )
        n = mat.shape[0]
        node_names = [f"node_{i}" for i in range(n)]
        return mat, node_names
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            "Please provide a .csv, .tsv, or .npy file."
        )


def main():
    args = parse_args()

    in_path = os.path.abspath(args.input)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    print(f"=== SPSID Denoising Interface ===")
    print(f"Input file : {in_path}")

    W_obs, node_names = load_matrix(in_path)
    print(f"Matrix shape: {W_obs.shape[0]} x {W_obs.shape[1]}")

    # 运行 SPSID（对一般用户，return_tf_only=False 更直观）
    W_denoised = spsid(
        W_obs,
        eps1=args.eps1,
        eps2=args.eps2,
        lambda_val=args.lambda_val,
        return_tf_only=False
    )

    # 构造输出路径
    if args.output is None:
        root, _ = os.path.splitext(in_path)
        out_path = root + "_SPSID.csv"
    else:
        out_path = os.path.abspath(args.output)

    df_out = pd.DataFrame(W_denoised, index=node_names, columns=node_names)
    df_out.to_csv(out_path)

    print(f"Output file: {out_path}")
    print(f"Denoising finished. Shape: {df_out.shape[0]} x {df_out.shape[1]}")


if __name__ == "__main__":
    main()
