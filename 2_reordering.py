import argparse
import os

import bintable
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input-data", required=True)
parser.add_argument("--input-clusters", required=True)
parser.add_argument("--output-data", required=True)
parser.add_argument("--output-clusters", required=True)


def main() -> None:
    args = parser.parse_args()
    assert args.input_data.endswith("/bintable.json")
    assert args.input_clusters.endswith(".npy")
    assert args.output_data.endswith("/bintable.json")
    assert args.output_clusters.endswith(".npy")
    df = bintable.read(
        os.path.dirname(args.input_data),
        only_columns=[
            "SOURCE_ID_GAIA",
            "scaled_En",
            "scaled_Lz",
            "scaled_Lperp",
        ],
    )
    Z = np.load(args.input_clusters, mmap_mode="r")
    assert Z.ndim == 2
    assert Z.shape[1] == 4

    output_points: list[int] = []
    output_clusters: list[tuple[int, int, float, int, int]] = []
    n = len(df)
    assert len(Z) == n - 1

    stack = [(-1, n - 2)]
    results: list[int] = []
    while stack:
        x, i = stack.pop()
        a = int(Z[i, 0])
        b = int(Z[i, 1])
        dist = Z[i, 2]
        sz = int(Z[i, 3])
        if x == -1:
            x = len(output_points)
            stack.append((x, i))
            if b < n:
                results.append(len(output_points))
                output_points.append(b)
            else:
                stack.append((-1, b - n))
            if a < n:
                results.append(len(output_points))
                output_points.append(a)
            else:
                stack.append((-1, a - n))
        else:
            y = len(output_points)
            assert y - x == sz, (x, y, sz)
            if b >= n:
                b = results.pop() + n
                assert output_clusters[b - n][4] == y
            else:
                b = results.pop()
            if a >= n:
                a = results.pop() + n
                assert output_clusters[a - n][3] == x
            else:
                a = results.pop()
            results.append(len(output_clusters))
            output_clusters.append((a, b, dist, x, y))

    assert len(df[output_points]) == n
    bintable.write(df[output_points], os.path.dirname(args.output_data))
    np.save(args.output_clusters, output_clusters)


if __name__ == "__main__":
    main()
