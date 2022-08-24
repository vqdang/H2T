# %%
import itertools
from collections import OrderedDict

from typing import Callable, Mapping, Union
import numpy as np
import sklearn
from scipy.spatial import Delaunay, Voronoi
from sklearn.neighbors import KDTree

from h2t.extract.features.abc import ABCFeatures


def find_y_in_x(x, y):
    """Return idx of element in `x` that appears in `y`."""
    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y
    result = np.ma.array(yindex, mask=mask)
    return result


def polygon_area(points):
    """Shoelace algorithm.
    Args:
        points (np.ndarray): Array of coordinates of
            shape `(N,2)` in `(x,y)` form.
    """
    x = points[:, 0]
    y = points[:, 1]
    a = np.dot(x, np.roll(y, 1))
    b = np.dot(y, np.roll(x, 1))
    return 0.5 * np.abs(a - b)


class KNNFeatures(ABCFeatures):
    def __init__(
        self,
        unique_type_list: Union[list, np.ndarray],
        pair_list: Union[list, np.ndarray] = None,
        kdtree: sklearn.neighbors.KDTree = None,
        compute_basic_stats: Callable[
            [Union[list, np.ndarray]], Mapping[str, str]
        ] = None,
    ):
        """
        `unique_type_list`: will be used to generate pair of typing for
            computing colocalization. This will explode if there are too
            many !

        """
        super().__init__()
        self.kdtree = kdtree
        self.pair_list = pair_list
        self.type_list = None  # holder for runtime
        self.unique_type_list = unique_type_list
        if compute_basic_stats is not None:
            self._compute_basic_stats = compute_basic_stats
        return

    def _get_neighborhood_stat(self, neighbor_idx_list):
        nn_type_list = self.type_list[neighbor_idx_list]
        (
            unique_type_list,
            nn_type_frequency,
        ) = np.unique(nn_type_list, return_counts=True)
        # repopulate for information wrt provided type because the
        # neighborhood may not contain all possible types within the
        # image/subject/dataset
        unique_type_freqency = np.zeros(len(self.unique_type_list))
        unique_type_freqency[unique_type_list] = nn_type_frequency
        return unique_type_freqency

    def transform(self, pts_list: np.ndarray, type_list: np.ndarray, radius: float):
        """Extract feature from a given coordinate list.

        During calculation, any node that has no neighbor will be
        excluded from calculating summarized statistics.

        `pts_list`: Nx2 array of coordinates. In case `kdtree` is provided
            when creating the object, `pts_list` input here must be the same
            one used to construct `kdtree` else the results will be bogus.
        `type_list`: N array indicating the type at each coordinate.

        """
        self.type_list = type_list
        kdtree = self.kdtree
        if kdtree is None:
            kdtree = KDTree(pts_list, metric="euclidean")
        knn_list = kdtree.query_radius(
            pts_list, radius, return_distance=True, sort_results=True
        )
        # each neighbor will contain self, so must remove self idx later
        knn_list = list(zip(*knn_list))

        # each clique is the neigborhood within the `radius` of point at
        # idx within `pts_list`
        keep_idx_list = []
        nn_raw_freq_list = []  # counnting occurence
        for idx, clique in enumerate(knn_list):
            nn_idx, nn_dst = clique
            if len(nn_idx) <= 1:
                continue
            # remove self to prevent bogus
            nn_idx = nn_idx[nn_idx != idx]
            nn_type_freq = self._get_neighborhood_stat(nn_idx)
            nn_raw_freq_list.append(nn_type_freq)
            keep_idx_list.append(idx)
        keep_idx_list = np.array(keep_idx_list)

        # from counting to ratio, may not entirely make sense
        # on the entire dataset
        def count_to_frequency(a):
            """Helper to make func less long."""
            return a / (np.sum(a, keepdims=True, axis=-1) + 1.0e-8)

        nn_raw_freq_list = np.array(nn_raw_freq_list)
        nn_rat_freq_list = count_to_frequency(nn_raw_freq_list)

        # get summarized occurence of pair of types, given the source
        # and its neighborhood
        pair_list = self.pair_list
        if pair_list is None:
            pair_list = itertools.product(self.unique_type_list, repeat=2)
            pair_list = list(pair_list)

        np.set_printoptions(precision=4, linewidth=160)
        stat_dict = OrderedDict()
        type_list = self.type_list[keep_idx_list]

        colocal_matrix = []
        for src_type in self.unique_type_list:
            sel = type_list == src_type
            src_nn_rat_list = (
                nn_rat_freq_list[sel]
                if np.sum(sel) > 0
                else np.zeros([1, len(self.unique_type_list)])
            )
            src_nn_avg_rat_list = np.mean(src_nn_rat_list, axis=0)
            colocal_matrix.append(src_nn_avg_rat_list)
        colocal_matrix = np.array(colocal_matrix)
        return colocal_matrix


class DelaunayFeatures(ABCFeatures):
    def transform(self, pts_list: np.ndarray):
        def find_delaunay_neighbors(graph):
            """
            Mainly to convert the .vertex_neighbor_vertices
            to a form that is more sensible to access. By default,
            vertex_neighbor_vertices from scipy contains 2 variables
            `indptr` and `indices`, which is flatten neighbors.
            For example, for `pts_list` of size 5
            >>> indptr = [ 0,  4,  7,  9, 12, 14]
            >>> indices = [2, 1, 3, 4, 0, 2, 3, 0, 1, 0, 1, 4, 0, 3]
            means
            `indices[0:4]` are indices of neighbor of pts_list[0],
            `indices[4:7]` are indices of neighbor of pts_list[1],
            `indices[12:14]` are indices of neighbor of pts_list[4].
            etc.
            """
            num_points = len(pts_list)

            nn_list = []
            idx_ptr, idx_list = graph.vertex_neighbor_vertices
            for vertex_idx in range(num_points):
                ptr_start = idx_ptr[vertex_idx]
                ptr_end = idx_ptr[vertex_idx + 1]
                nn_idx_list = idx_list[ptr_start:ptr_end]
                nn_list.append(nn_idx_list)
            return nn_list

        # delaunay is in YX not XY
        graph = Delaunay(pts_list)
        nn_list = find_delaunay_neighbors(graph)
        num_neighbor_per_vertex_list = [len(v) for v in nn_list]
        # find the number of triangles for each vertex
        # ! do not use vertex_to_simplex because it only contains
        # ! partial number of simplices per vertex

        # simplices are the triangle list, each triangle is a list
        # of idx of `pts_list`, so counting the frequency of each idx
        # should provide the number of triangles surronding each vertex
        (_, num_triangles_per_vertex_list) = np.unique(
            graph.simplices, return_counts=True
        )

        area_list = []
        for idx_list in graph.simplices:
            vertex_list = pts_list[idx_list]
            area = polygon_area(vertex_list)
            area_list.append(area)
        # calculate basic statistics
        stat_dict = OrderedDict()
        stat_dict["Number of Delaunay triangles"] = len(graph.simplices)

        # ! can do sthg about these duplications?
        desc = f"Number of Neighbors per Delaunay Vertex"
        xstat_dict = self._compute_basic_stats(num_neighbor_per_vertex_list)
        xstat_dict = {f"[{k}] {desc}": v for k, v in xstat_dict.items()}
        stat_dict.update(xstat_dict)

        desc = f"Number of Triangles per Delaunay Vertex"
        xstat_dict = self._compute_basic_stats(num_triangles_per_vertex_list)
        xstat_dict = {f"[{k}] {desc}": v for k, v in xstat_dict.items()}
        stat_dict.update(xstat_dict)

        desc = f"Area of Delaunay Triangle"
        xstat_dict = self._compute_basic_stats(area_list)
        xstat_dict = {f"[{k}] {desc}": v for k, v in xstat_dict.items()}
        stat_dict.update(xstat_dict)

        return stat_dict


class VoronoiFeature(ABCFeatures):
    def transform(self, pts_list: np.ndarray):
        graph = Voronoi(pts_list)
        # exclude voronoi region going toward infiniy,
        # region going to infinity will vertex with idx=-1
        roi_list = graph.regions
        roi_list = [v for v in roi_list if np.all(np.array(v) > 0)]

        area_list = []
        for roi_idx, idx_list in enumerate(roi_list):
            vertex_list = graph.vertices[idx_list]
            area = polygon_area(vertex_list)
            area_list.append(area)

        # calculate basic statistics
        stat_dict = OrderedDict()
        stat_dict["Number of Voronoi Regions"] = len(roi_list)

        desc = f"Area of Voronoi Regions"
        xstat_dict = self._compute_basic_stats(area_list)
        xstat_dict = {f"[{k}] {desc}": v for k, v in xstat_dict.items()}
        stat_dict.update(xstat_dict)

        return stat_dict
