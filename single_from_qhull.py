import numpy as np
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay


def single_linkage(points):
    # Compute Euclidean MST using Delaunay triangulation, then run Union-Find.
    return single_linkage_from_mst(euclidean_mst_scipy(points))


def single_linkage_quadratic_time(points):
    # Compute Euclidean MST using a quadratic algorithm, then run Union-Find.
    return single_linkage_from_mst(euclidean_mst_quadratic(points))


def euclidean_mst_scipy(points):
    # Run Delaunay triangulation to obtain a suitable input for Euclidean MST.
    tri = Delaunay(points)
    u, v = np.transpose(
        sorted(
            {
                (i, j)
                for sim in tri.simplices
                for ind, i in enumerate(sim)
                for j in sim[ind + 1 :]
            }
        )
    )
    # Obtain relevant Euclidean distances for Euclidean MST.
    data = ((points[u] - points[v]) ** 2).sum(axis=1) ** 0.5

    # Run MST on Euclidean-weighted sparse graph.
    X = coo_array((data, (u, v)), shape=(len(points), len(points)))
    return coo_array(minimum_spanning_tree(csr_matrix(X)))


def euclidean_mst_quadratic(points):
    # Prim's algorithm to compute a minimum spanning tree:
    # Initialize the MST with a single arbitrary input point.
    # Repeat the following N-1 times:
    # Identify the closest point not already added to the MST,
    # and add it to the MST.
    # Usually this would be implemented with an efficient priority queue,
    # leading to an N log N runtime.
    # But this is Python, so our priority queue is a simple unsorted array,
    # leading to quadratic runtime.
    data = []
    u = []
    v = []

    # non-nan dists[i] is the Euclidean distance between i and closest[i]
    # if dists[i] is nan, i has already been added to the MST
    dists = ((points[0:1] - points) ** 2).sum(axis=1) ** 0.5
    dists[0] = np.nan
    closest = np.array([0] * len(points), dtype=np.intp)
    for _ in range(len(points)):
        # Add the closest not-yet-added point into the MST
        try:
            i = np.nanargmin(dists)
        except ValueError:
            # All points added
            break
        # Add to MST
        data.append(dists[i])
        u.append(i)
        v.append(closest[i])
        # Update dists and closest with distances to i
        newdists = ((points[i : i + 1] - points) ** 2).sum(axis=1) ** 0.5
        newclosest = newdists < dists
        closest[newclosest] = i
        np.minimum(dists, newdists, out=dists)
        # Mark i as visited
        dists[i] = np.nan
    # After n-1 iterations, all should be visited, so all of dists should be nan
    assert np.all(np.isnan(dists))
    return coo_array((data, (u, v)), shape=(len(points), len(points)))


def single_linkage_from_mst(Tcsr):
    # Run Union-Find on MST to obtain single-linkage clustering.
    Z: list[tuple[int, int, float, int]] = []
    n: int = Tcsr.shape[0]
    parent: list[int] = list(range(n))
    subtree_size: list[int] = [1] * n

    for i in np.argsort(Tcsr.data):
        u = Tcsr.row[i]
        v = Tcsr.col[i]
        dist = Tcsr.data[i]
        while parent[u] != u:
            parent[u] = u = parent[parent[u]]
        while parent[v] != v:
            parent[v] = v = parent[parent[v]]
        my_size = subtree_size[u] + subtree_size[v]
        parent[u] = parent[v] = len(parent)
        parent.append(len(parent))
        subtree_size.append(my_size)
        if u > v:
            u, v = v, u
        Z.append((u, v, dist, my_size))

    return np.array(Z)


if __name__ == "__main__":
    main()
