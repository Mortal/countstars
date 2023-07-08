import copy
import os
import pickle

import bintable
import numpy as np
import scipy.linalg
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


def main() -> None:
    df = bintable.read("df_reordered")
    Z = np.load("df_reordered_clusters.npy", mmap_mode="r")
    assert Z.ndim == 2
    n = len(df)
    assert Z.shape[0] == n - 1
    assert Z.shape[1] == 5

    # data = np.hstack(df["scaled_En", "scaled_Lz", "scaled_Lperp"].columns.values())
    # pca = IncrementalPCA().fit(data)
    # # [-1.32446951 -0.89720671 -0.9997682 ] [0.99894886 1.01436526 0.91940532]
    # print(data.min(axis=0), data.max(axis=0))

    # N_sigma_ellipse_axis = 2.83 #Length (in standard deviations) of axis of ellipsoidal cluster boundary
    # n_std = N_sigma_ellipse_axis
    # box = pca.inverse_transform(np.r_[n_std ** 2 * np.eye(3), n_std ** 2 * -np.eye(3)])
    # # [-8.17079986 -7.93803492 -8.26954146] [6.81314303 7.95731164 6.69840794]
    # print(box.min(axis=0), box.max(axis=0))

    # return

    def get_range(i):
        if i < n:
            return i, i + 1
        return int(Z[i - n, 3]), int(Z[i - n, 4])

    work = []
    for i in range(len(Z)):
        a = int(Z[i, 0])
        b = int(Z[i, 1])
        dist = Z[i, 2]
        x = int(Z[i, 3])
        y = int(Z[i, 4])
        ax, ay = get_range(a)
        bx, by = get_range(b)
        assert ax == x, (i, n, a, b, ax, ay, bx, by, x, y)
        assert ay == bx
        assert by == y
        asize = ay - ax
        bsize = by - bx
        mysize = y - x
        if mysize < 5:
            work.append((-2, x, y))
        elif mysize < 10:
            work.append((-1, x, y))
        elif asize > bsize:
            assert asize > 1
            work.append((a - n, bx, by))
        else:
            assert bsize > 1
            work.append((b - n, ax, ay))

    pcas: list[IncrementalPCA | None] = []
    t = tqdm(total=sum(y - x for i, x, y in work if i != -2))
    for i, x, y in work:
        assert isinstance(x, int), x
        assert isinstance(y, int), y
        data = np.hstack(
            df[x:y]["scaled_En", "scaled_Lz", "scaled_Lperp"].columns.values()
        )
        if i == -2:
            pcas.append(None)
        elif i == -1:
            assert data.shape[1] == 3
            assert data.shape[0] > 3
            pca = IncrementalPCA(n_components=3, whiten=True).fit(data)
            pcas.append(pca)
            assert pca.n_components_ == 3
            assert pca.n_features_in_ == 3
            assert pca._n_features_out == 3, pca._n_features_out
            assert pca.components_.shape == (3, 3), pca.components_.shape
            t.update(y - x)
        else:
            src = pcas[i]
            assert src is not None
            pca = copy.deepcopy(src).partial_fit(data)
            pcas.append(pca)
            assert pca.n_components_ == 3
            assert pca.components_.shape == (3, 3)
            t.update(y - x)
    t.close()
    with open("pcas.pkl_", "wb") as fp:
        pickle.dump(pcas, fp)
    os.rename("pcas.pkl_", "pcas.pkl")


if __name__ == "__main__":
    main()
