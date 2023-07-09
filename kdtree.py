from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.decomposition import IncrementalPCA


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

