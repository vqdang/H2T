import numpy as np

from sklearn.utils import check_array

from sklearn.cluster import MiniBatchKMeans

from sklearn.cluster._kmeans import _mini_batch_step, _labels_inertia_threadpool_limit
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import threadpool_limits
from sklearn.utils.validation import _check_sample_weight, check_random_state

from sklearn.decomposition import MiniBatchDictionaryLearning


class ExtendedMiniBatchKMeans(MiniBatchKMeans):
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

    def prototypical_patterns(self):
        return np.copy(self.cluster_centers_)


class ExtendedMiniBatchDictionaryLearning(MiniBatchDictionaryLearning):
    def partial_fit(
        self,
        X,
        y=None,
        # Additional arguments to calculate the
        # convergence condition in-place
        # step, n_steps, n_samples = check_convergence
        check_convergence=None,
    ):
        """Update dictionary based on a single mini-batch X.

        This extends the original `MiniBatchDictionaryLearning` to include
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
        if check_convergence is not None and check_convergence[0] == 0:
            # Attributes to monitor the convergence
            self._ewa_cost = None
            self._ewa_cost_min = None
            self._no_improvement = 0

        has_components = hasattr(self, "components_")

        X = self._validate_data(
            X, dtype=[np.float64, np.float32], order="C", reset=not has_components
        )

        self.n_steps_ = getattr(self, "n_steps_", 0)

        if not has_components:
            # This instance has not been fitted yet (fit or partial_fit)
            self._check_params(X)
            self._random_state = check_random_state(self.random_state)

            dictionary = self._initialize_dict(X, self._random_state)

            self._inner_stats = (
                np.zeros((self._n_components, self._n_components), dtype=X.dtype),
                np.zeros((X.shape[1], self._n_components), dtype=X.dtype),
            )
        else:
            dictionary = self.components_

        batch_cost = self._minibatch_step(
            X, dictionary, self._random_state, self.n_steps_
        )

        self.components_ = dictionary
        self.n_steps_ += 1

        # Monitor convergence and do early stopping if necessary
        if check_convergence is not None:
            step, n_steps, n_samples = check_convergence
            old_dictionary = dictionary
            new_dictionary = self.components_
            # print(old_dictionary)
            # print(new_dictionary)
            is_converge = self._check_convergence(
                X, batch_cost, new_dictionary, old_dictionary, n_samples, step, n_steps
            )
            return self, is_converge
        return self

    def prototypical_patterns(self):
        return np.copy(self.components_)


if __name__ == "__main__":
    """Local code to test implementations"""

    X = np.array(
        [
            [1, 2],
            [1, 4],
            [1, 0],
            [4, 2],
            [4, 0],
            [4, 4],
            [4, 5],
            [0, 1],
            [2, 2],
            [3, 2],
            [5, 5],
            [1, -1],
        ]
    )

    # manually fit on batches
    num_steps = 128
    num_samples = X.shape[0]
    batch_size = 6
    seed = 5
    np.random.seed(seed)

    model_name = "dictionary"
    if model_name == "kmean":
        model = ExtendedMiniBatchKMeans(
            n_clusters=2, random_state=0, batch_size=batch_size
        )
        base_model = MiniBatchKMeans(
            n_clusters=2, random_state=0, batch_size=batch_size
        )
    elif model_name == "dictionary":
        model = ExtendedMiniBatchDictionaryLearning(
            n_components=2, random_state=0, batch_size=6
        )
        base_model = MiniBatchDictionaryLearning(
            n_components=2, random_state=0, batch_size=6
        )

    for step in range(num_steps):
        x_batch = np.random.randint(0, num_samples, batch_size)
        x_batch = X[x_batch]
        check_convergence = [step, num_steps, num_samples]
        model, is_converged = model.partial_fit(
            x_batch, check_convergence=check_convergence
        )
        base_model = base_model.partial_fit(x_batch)
        if model_name == "kmean":
            print(
                is_converged,
                np.sum(base_model.cluster_centers_ - model.cluster_centers_),
            )
        elif model_name == "dictionary":
            print(is_converged, np.sum(base_model.components_ - model.components_))

    print("here")
