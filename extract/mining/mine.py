
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class MiniBatchKMeansExtended(MiniBatchKMeans):
    def partial_fitX(
            self, X, iter, n_iter, n_samples,
            y=None, sample_weight=None):
        """Update k means estimate on a single mini-batch X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Coordinates of the data points to cluster. It must be noted that
            X will be copied if it is not C-contiguous.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).

        Returns
        -------
        self
        """
        stop = False
        is_first_call_to_partial_fit = iter == 0

        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', accept_large_sparse=False,
                                reset=is_first_call_to_partial_fit)

        self._random_state = getattr(self, "_random_state",
                                        check_random_state(self.random_state))
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        x_squared_norms = row_norms(X, squared=True)

        if is_first_call_to_partial_fit:
            # this is the first call to partial_fit on this object
            self._check_params(X)

            # Validate init array
            init = self.init
            if hasattr(init, '__array__'):
                init = check_array(init, dtype=X.dtype, copy=True, order='C')
                self._validate_center_shape(X, init)

            self._check_mkl_vcomp(X, X.shape[0])

            # initialize the cluster centers
            self.cluster_centers_ = self._init_centroids(
                X, x_squared_norms=x_squared_norms,
                init=init,
                random_state=self._random_state,
                init_size=self._init_size)

            self._counts = np.zeros(self.n_clusters,
                                    dtype=sample_weight.dtype)
            self._convergence_context = {}

        distances = np.zeros(X.shape[0], dtype=X.dtype)
        random_reassign=((iter + 1) % (10 + int(self._counts.min())) == 0)
        # Perform the actual update step on the minibatch data
        batch_inertia, centers_squared_diff = _mini_batch_step(
            X, sample_weight, x_squared_norms,
            self.cluster_centers_, self.counts_,
            np.zeros(0, dtype=X.dtype), 0,
            random_reassign=random_reassign, distances=distances,
            random_state=self._random_state,
            reassignment_ratio=self.reassignment_ratio,
            verbose=self.verbose)

        # Monitor convergence and do early stopping if necessary
        if _mini_batch_convergence(
                self, iter, n_iter,
                False, n_samples,
                centers_squared_diff,
                batch_inertia,
                self._convergence_context,
                verbose=self.verbose):
            stop = True

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia(
                X, sample_weight, x_squared_norms, self.cluster_centers_)

        return self, stop