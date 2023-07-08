import bintable
import numpy as np


def main() -> None:
    df = bintable.read(
        "df",
        only_columns=[
            "SOURCE_ID_GAIA",
            "scaled_En",
            "scaled_Lz",
            "scaled_Lperp",
        ],
    )
    Z = np.load("Z.npy", mmap_mode="r")

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
    bintable.write(df[output_points], "df_reordered")
    np.save("df_reordered_clusters.npy", output_clusters)


if __name__ == "__main__":
    main()
