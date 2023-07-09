import argparse
import os
import pickle

import bintable
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from kdtree import build_kdtree, visit_kdtree, PointInsideCounter


parser = argparse.ArgumentParser()
parser.add_argument("--input-pcas", required=True)
parser.add_argument("--input-data", required=True)
parser.add_argument("--output-counts", required=True)


def main() -> None:
    args = parser.parse_args()
    assert args.input_pcas.endswith(".pkl")
    assert args.input_data.endswith("/bintable.json")
    assert args.output_counts.endswith(".npy")
    with open(args.input_pcas, "rb") as fp:
        pcas: list[IncrementalPCA | None] = pickle.load(fp)
    assert isinstance(pcas, list)
    assert all(pca is None or isinstance(pca, IncrementalPCA) for pca in pcas)
    pca = next(pca for pca in pcas if pca is not None)
    n_components, n_features = pca.components_.shape
    assert n_components == 3, n_components
    assert n_features == 3
    assert all(
        pca is None or pca.components_.shape == (n_components, n_features)
        for pca in pcas
    )
    df = bintable.read(os.path.dirname(args.input_data))
    # Z = np.load("df_reordered_clusters.npy")
    data = np.hstack(df["scaled_En", "scaled_Lz", "scaled_Lperp"].columns.values())

    # Length (in standard deviations) of axis of ellipsoidal cluster boundary
    N_sigma_ellipse_axis = 2.83
    n_std = N_sigma_ellipse_axis

    output = []
    pcas2 = [(i, pca) for i, pca in enumerate(pcas) if pca is not None]

    if args.no_kdtree:
        for i, pca in tqdm(pcas2):
            assert pca is not None
            box = pca.inverse_transform(
                np.r_[n_std ** 2 * np.eye(3), n_std ** 2 * -np.eye(3)]
            )
            boxmin = box.min(axis=0, keepdims=True)
            boxmax = box.max(axis=0, keepdims=True)
            boxfilt = np.all((boxmin <= data) & (data <= boxmax), axis=1)
            transformed = pca.transform(data[boxfilt])
            filt = (transformed ** 2).sum(axis=1) <= n_std ** 2
            output.append((i, np.sum(filt)))
    else:
        tree = build_kdtree(data)
        for i, pca in tqdm(pcas2):
            assert pca is not None
            ctr = PointInsideCounter(pca=pca, n_std=n_std)
            visit_kdtree(tree, ctr)
            output.append((i, ctr.inside))

    np.save(args.output_counts, output, allow_pickle=False)


if __name__ == "__main__":
    main()
