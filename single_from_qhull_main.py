import time

import numpy as np
import scipy.cluster.hierarchy
from sklearn.cluster import AgglomerativeClustering  # type: ignore[import]

from single_from_qhull import single_linkage, single_linkage_quadratic_time


def main() -> None:
    seed = 42
    np.random.seed(seed)
    pts = np.random.random((20, 10))
    print(f"Random points (seed={seed}):")
    print(pts)
    a = single_linkage(pts)
    print("\nOutput of single_linkage():")
    print(a)
    b = scipy.cluster.hierarchy.linkage(pts)
    print("\nOutput of scipy.cluster.hierarchy.linkage():")
    print(b)
    assert np.allclose(a, b)

    np.random.seed(seed)
    pts = np.random.random((10000, 2))
    t1 = time.time()
    print(single_linkage(pts)[-1])
    t2 = time.time()
    print(f"Time to run N log N time algorithm on {pts.shape[0]} {pts.shape[1]}-d points: {t2 - t1} s")

    np.random.seed(seed)
    pts = np.random.random((10000, 2))
    t1 = time.time()
    print(single_linkage_quadratic_time(pts)[-1])
    t2 = time.time()
    print(f"Time to run N² time algorithm on {pts.shape[0]} {pts.shape[1]}-d points: {t2 - t1} s")

    np.random.seed(seed)
    pts = np.random.random((100000, 2))
    t1 = time.time()
    print(single_linkage(pts)[-1])
    t2 = time.time()
    print(f"Time to run N log N time algorithm on {pts.shape[0]} {pts.shape[1]}-d points: {t2 - t1} s")

    for n in (10000, 20000, 40000):
        np.random.seed(seed)
        pts = np.random.random((n, 2))
        t1 = time.time()
        print(AgglomerativeClustering(linkage="single").fit(pts))
        t2 = time.time()
        print(f"Time to run AgglomerativeClustering time algorithm on {pts.shape[0]} {pts.shape[1]}-d points: {t2 - t1} s")


if __name__ == "__main__":
    main()

