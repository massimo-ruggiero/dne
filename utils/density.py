from sklearn.covariance import LedoitWolf, empirical_covariance
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
import torch
import numpy as np

class Density(object):
    def fit(self, embeddings):
        raise NotImplementedError

    def predict(self, embeddings):
        raise NotImplementedError


def is_gmm_density(density):
    return isinstance(density, GMMDensitySklearn)


def is_gmm_committee(density):
    return isinstance(density, GMMCommitteeDensity)


def get_density(args):
    density_cfg = getattr(args, "density", None)
    density_name = getattr(density_cfg, "name", "gde")

    if density_name == "gmm":
        gmm_cfg = getattr(density_cfg, "gmm", None)
        n_components = getattr(gmm_cfg, "n_components", 3)
        reg_covar = getattr(gmm_cfg, "reg_covar", 1e-6)
        covariance_type = getattr(gmm_cfg, "covariance_type", "full")
        max_iter = getattr(gmm_cfg, "max_iter", 100)
        random_state = getattr(args, "seed", 42)
        return GMMDensitySklearn(
            n_components=n_components,
            reg_covar=reg_covar,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
        )
    if density_name == "gmm_committee":
        gmm_cfg = getattr(density_cfg, "gmm", None)
        n_components = getattr(gmm_cfg, "n_components", 3)
        reg_covar = getattr(gmm_cfg, "reg_covar", 1e-5)
        covariance_type = getattr(gmm_cfg, "covariance_type", "full")
        max_iter = getattr(gmm_cfg, "max_iter", 100)
        random_state = getattr(args, "seed", 42)
        return GMMCommitteeDensity(
            n_components=n_components,
            reg_covar=reg_covar,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=random_state,
        )
    if density_name == "kde":
        return GaussianDensitySklearn()
    return GaussianDensityTorch()


class GaussianDensityTorch(object):
    """Gaussian Density estimation similar to the implementation used by Ripple et al.
    The code of Ripple et al. can be found here: https://github.com/ORippler/gaussian-ad-mvtec.
    """

    def fit(self, embeddings):
        self.mean = torch.mean(embeddings, dim=0)
        # self.cov = torch.Tensor(empirical_covariance(embeddings - self.mean), device="cpu")
        self.cov = torch.Tensor(LedoitWolf().fit(embeddings.cpu()).covariance_, device="cpu")
        self.inv_cov = torch.Tensor(LedoitWolf().fit(embeddings.cpu()).precision_, device="cpu")
        return self.mean, self.cov

    def predict(self, embeddings):
        distances = self.mahalanobis_distance(embeddings, self.mean, self.inv_cov)
        return distances

    @staticmethod
    def mahalanobis_distance(
            values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.
        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.

        from https://github.com/ORippler/gaussian-ad-mvtec/blob/4e85fb5224eee13e8643b684c8ef15ab7d5d016e/src/gaussian/model.py#L308
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert len(inv_covariance.shape) == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        return dist.sqrt()


class GaussianDensitySklearn():
    """Li et al. use sklearn for density estimation.
    This implementation uses sklearn KernelDensity module for fitting and predicting.
    """

    def fit(self, embeddings):
        # estimate KDE parameters
        # use grid search cross-validation to optimize the bandwidth
        self.kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(embeddings)

    def predict(self, embeddings):
        scores = self.kde.score_samples(embeddings)

        # invert scores, so they fit to the class labels for the auc calculation
        scores = -scores

        return scores


class GMMDensitySklearn():
    def __init__(self, n_components, reg_covar, covariance_type='full', random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.gmm  = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state
        )
        self.is_fitted = False
        
    def fit(self, embeddings):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        self.gmm.fit(embeddings)
        self.is_fitted = True
        return self.gmm.means_, self.gmm.covariances_

    def predict(self, embeddings):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        scores = -self.gmm.score_samples(embeddings)
        
        return scores
    
    def sample(self, n_samples):
        if not self.is_fitted:
            raise RuntimeError("GMM not fitted. Call fit() first.")
        
        samples, _ = self.gmm.sample(n_samples)
        return samples


class GMMCommitteeDensity():
    def __init__(self, n_components, reg_covar, covariance_type='full', max_iter=100, random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state
        self.memory_list = []
        self.current_record = None

    def fit_task(self, embeddings, task_id=None, save=True):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        gmm.fit(embeddings)
        record = {
            "task_id": task_id,
            "means": gmm.means_,
            "covariances": gmm.covariances_,
            "weights": gmm.weights_,
            "precisions_cholesky": gmm.precisions_cholesky_,
        }
        if save:
            self.memory_list.append(record)
            self.current_record = None
        else:
            self.current_record = record
        return gmm.means_, gmm.covariances_

    def predict(self, embeddings):
        records = list(self.memory_list)
        if self.current_record is not None:
            records.append(self.current_record)
        if len(records) == 0:
            raise RuntimeError("GMM committee is empty. Call fit_task() first.")
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        log_likelihoods = []
        for record in records:
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            gmm.means_ = record["means"]
            gmm.covariances_ = record["covariances"]
            gmm.weights_ = record["weights"]
            gmm.precisions_cholesky_ = record["precisions_cholesky"]
            log_likelihoods.append(gmm.score_samples(embeddings))
        log_likelihoods = np.stack(log_likelihoods, axis=1)  # [n_samples, n_models]
        max_log_likelihood = np.max(log_likelihoods, axis=1)
        return -max_log_likelihood
