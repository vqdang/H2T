import numpy as np
from sklearn.utils import check_array
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster._kmeans import _mini_batch_step, _labels_inertia_threadpool_limit
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import threadpool_limits
from sklearn.utils.validation import _check_sample_weight, check_random_state


class MiniBatchKMeansExtended(MiniBatchKMeans):
    def partial_fit(
        self,
        X,
        y=None,
        sample_weight=None,
        # Additional arguments to calculate the
        # convergence condition in-place
        # step, n_steps, n_samples = check_convergence
        check_convergence=None,
    ):
        """Update k means estimate on a single mini-batch X.

        This extends the original `MiniBatchKMeans` to include
        convergence checking with respect to external step metadata.

        Args:
            check_convergence (tuple): A tuple that contains the current
                step `partial_fit` metadata. It is expected to be of the
                form `check_convergence = (step, n_steps, n_samples)` where

                - `step` is the index of the current optimization step (or n-th
                call of `partial_fit`).

                - `n_steps` is the total number of optimization step to be done.

                - `n_samples`: By denoting the data consumed by each `partial_fit`
                call a mini batch, `n_samples` denote the total number of sample
                in the dataset where these mini batches come from.

                By default, `check_convergence = None` and the checking is not
                performed.

        """
        has_centers = hasattr(self, "cluster_centers_")

        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
            reset=not has_centers,
        )

        self._random_state = getattr(
            self, "_random_state", check_random_state(self.random_state)
        )
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self.n_steps_ = getattr(self, "n_steps_", 0)

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if not has_centers:
            # this instance has not been fitted yet (fit or partial_fit)
            self._check_params(X)
            self._n_threads = _openmp_effective_n_threads()

            # Validate init array
            init = self.init
            if hasattr(init, "__array__"):
                init = check_array(init, dtype=X.dtype, copy=True, order="C")
                self._validate_center_shape(X, init)

            self._check_mkl_vcomp(X, X.shape[0])

            # initialize the cluster centers
            self.cluster_centers_ = self._init_centroids(
                X,
                x_squared_norms=x_squared_norms,
                init=init,
                random_state=self._random_state,
                init_size=self._init_size,
            )

            # Initialize counts
            self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

            # Initialize number of samples seen since last reassignment
            self._n_since_last_reassign = 0

        with threadpool_limits(limits=1, user_api="blas"):
            old_centers = np.copy(self.cluster_centers_)
            batch_inertia = _mini_batch_step(
                X,
                x_squared_norms=x_squared_norms,
                sample_weight=sample_weight,
                centers=self.cluster_centers_,
                centers_new=self.cluster_centers_,
                weight_sums=self._counts,
                random_state=self._random_state,
                random_reassign=self._random_reassign(),
                reassignment_ratio=self.reassignment_ratio,
                verbose=self.verbose,
                n_threads=self._n_threads,
            )

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X,
                sample_weight,
                x_squared_norms,
                self.cluster_centers_,
                n_threads=self._n_threads,
            )

        self.n_steps_ += 1
        # Monitor convergence and do early stopping if necessary
        if check_convergence is not None:
            step, n_steps, n_samples = check_convergence
            if step == 0:
                # Attributes to monitor the convergence
                self._ewa_inertia = None
                self._ewa_inertia_min = None
                self._no_improvement = 0

            new_centers = self.cluster_centers_
            centers_squared_diff = (
                np.sum((new_centers - old_centers) ** 2) if self._tol > 0.0 else 0.0
            )
            is_converge = self._mini_batch_convergence(
                step, n_steps, n_samples, centers_squared_diff, batch_inertia
            )
            return self, is_converge
        return self


if __name__ == "__main__":
    # X = np.array(
    #     [
    #         [1, 2],
    #         [1, 4],
    #         [1, 0],
    #         [4, 2],
    #         [4, 0],
    #         [4, 4],
    #         [4, 5],
    #         [0, 1],
    #         [2, 2],
    #         [3, 2],
    #         [5, 5],
    #         [1, -1],
    #     ]
    # )
    # # manually fit on batches
    # kmeans = MiniBatchKMeansExtended(n_clusters=2, random_state=0, batch_size=6)
    # check_convergence = [0, 5, X.shape[0]]
    # kmeans, is_converged = kmeans.partial_fit(X[0:6, :], check_convergence=check_convergence)
    # print(is_converged)
    # check_convergence = [1, 5, X.shape[0]]
    # kmeans, is_converged = kmeans.partial_fit(X[6:12, :], check_convergence=check_convergence)
    # print(is_converged)
    # check_convergence = [2, 5, X.shape[0]]
    # kmeans, is_converged = kmeans.partial_fit(X[4:10, :], check_convergence=check_convergence)
    # print(is_converged)

    # kmeans.cluster_centers_
    # kmeans.predict([[0, 0], [4, 4]])
    # # fit on the whole data
    # kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=6, max_iter=10).fit(X)
    # kmeans.cluster_centers_
    # kmeans.predict([[0, 0], [4, 4]])
    print("here")
