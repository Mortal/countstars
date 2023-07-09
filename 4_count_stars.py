import argparse
import collections
import os
import pickle
from dataclasses import dataclass
from typing import Callable

import bintable
import numpy as np
import scipy.spatial
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--input-pcas", required=True)
parser.add_argument("--input-data", required=True)
parser.add_argument("--output-counts", required=True)


@dataclass
class KdtreeNode:
    i: int
    j: int
    boxmin: np.ndarray
    boxmax: np.ndarray
    left: int
    right: int


@dataclass
class Kdtree:
    points: np.ndarray
    nodes: list[KdtreeNode]


def build_kdtree(points) -> Kdtree:
    nodes: list[KdtreeNode] = []
    n = len(points)

    def build(i: int, j: int) -> int:
        if i + 1 == j:
            return i
        boxmin = points[i:j].min(axis=0)
        boxmax = points[i:j].max(axis=0)
        box = boxmax - boxmin
        dim = np.argmax(box)
        half = (j - i) // 2
        part = np.argpartition(points[i:j, dim], half)
        points[i:j] = points[i:j][part]
        left = build(i, i + half)
        right = build(i + half, j)
        r = len(nodes)
        nodes.append(KdtreeNode(i, j, boxmin, boxmax, left, right))
        return r + n

    build(0, n)

    return Kdtree(points, nodes)


def visit_kdtree(
    tree: Kdtree, f: Callable[[np.ndarray, np.ndarray, np.ndarray], bool]
) -> None:
    n = len(tree.points)
    assert len(tree.nodes) == n - 1
    visit = [n + len(tree.nodes) - 1]
    visits = 0

    while visit:
        visits += 1
        i = visit.pop()
        if i < n:
            f(tree.points[i], tree.points[i], tree.points[i : i + 1])
            continue
        node = tree.nodes[i - n]
        if f(node.boxmin, node.boxmax, tree.points[node.i : node.j]):
            visit.append(node.right)
            visit.append(node.left)


class PointInsideCounter:
    def __init__(self, pca: IncrementalPCA, *, n_std: float) -> None:
        self.pca = pca
        self.n_std = n_std
        self.inside = 0
        box = pca.inverse_transform(
            np.r_[n_std ** 2 * np.eye(3), n_std ** 2 * -np.eye(3)]
        )
        self.boxmin = box.min(axis=0)
        self.boxmax = box.max(axis=0)

    def __call__(self, boxmin, boxmax, points) -> bool:
        overlap = (boxmin < self.boxmax) & (self.boxmin < boxmax)
        if not overlap.all():
            return False
        if len(points) < 50000:
            boxfilt = (
                (self.boxmin[None, :] <= points) & (points <= self.boxmax[None, :])
            ).all(axis=1)
            points_box = points[boxfilt]
            if len(points_box) > 0:
                points_transformed = self.pca.transform(points_box)
                points_filt = (points_transformed ** 2).sum(axis=1) <= self.n_std ** 2
                self.inside += np.sum(points_filt)
            return False
        corners = np.array(
            [
                [boxmin[0], boxmin[1], boxmin[2]],
                [boxmin[0], boxmin[1], boxmax[2]],
                [boxmin[0], boxmax[1], boxmin[2]],
                [boxmin[0], boxmax[1], boxmax[2]],
                [boxmax[0], boxmin[1], boxmin[2]],
                [boxmax[0], boxmin[1], boxmax[2]],
                [boxmax[0], boxmax[1], boxmin[2]],
                [boxmax[0], boxmax[1], boxmax[2]],
            ]
        )
        corners_transformed = self.pca.transform(corners)
        corners_filt = (corners_transformed ** 2).sum(axis=1) <= self.n_std ** 2
        if np.all(corners_filt):
            self.inside += len(points)
            return False
        return True


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
