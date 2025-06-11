import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class BayesianPMF:
    """
    Bayesian Probabilistic Matrix Factorization w posterior uncertainty
    Model:
    - R_ij ~ N(U_i^T V_j, o_r^2)
    - U_i ~ N(μ_u, Σ_u)
    - V_j ~ N(μ_v, Σ_v)
    """

    def __init__(self, n_factors=10, learning_rate=0.01, n_iterations=100,
                 prior_variance=1.0, noise_variance=1.0, verbose=True):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.prior_variance = prior_variance
        self.noise_variance = noise_variance
        self.verbose = verbose
        self.mu_u = None  # Mean of users
        self.sigma_u = None  # Variance of users
        self.mu_v = None  # Mean of items
        self.sigma_v = None  # Variance of items

    def fit(self, train_data, val_data=None):
        users = set([x[0] for x in train_data])
        jokes = set([x[1] for x in train_data])
        self.n_users = len(users)
        self.n_jokes = len(jokes)
        self.user_to_idx = {user: idx for idx, user in enumerate(sorted(users))}
        self.joke_to_idx = {joke: idx for idx, joke in enumerate(sorted(jokes))}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_joke = {idx: joke for joke, idx in self.joke_to_idx.items()}
        self._initialize_variational_parameters()
        train_indices = [(self.user_to_idx[u], self.joke_to_idx[j], r)
                        for u, j, r in train_data]
        self.train_losses = []
        self.val_losses = []
        self.elbo_history = []
        for iteration in range(self.n_iterations):
            np.random.shuffle(train_indices)
            elbo = self._update_variational_parameters(train_indices)
            self.elbo_history.append(elbo)
            train_loss = self.evaluate(train_data)
            self.train_losses.append(train_loss)
            if val_data:
                val_loss = self.evaluate(val_data)
                self.val_losses.append(val_loss)
                if self.verbose and iteration % 10 == 0:
                    print(f"Iteration {iteration}: ELBO = {elbo:.4f}, "
                          f"Train RMSE = {train_loss:.4f}, Val RMSE = {val_loss:.4f}")
            else:
                if self.verbose and iteration % 10 == 0:
                    print(f"Iteration {iteration}: ELBO = {elbo:.4f}, "
                          f"Train RMSE = {train_loss:.4f}")

    def _initialize_variational_parameters(self):
        # User:smaller initial values
        self.mu_u = np.random.normal(0, 0.01, (self.n_users, self.n_factors))
        self.sigma_u = np.ones((self.n_users, self.n_factors)) * 0.1

        # Item: smaller initial values
        self.mu_v = np.random.normal(0, 0.01, (self.n_jokes, self.n_factors))
        self.sigma_v = np.ones((self.n_jokes, self.n_factors)) * 0.1

    def _update_variational_parameters(self, train_indices):
        self.mu_u = np.clip(self.mu_u, -10, 10)
        self.mu_v = np.clip(self.mu_v, -10, 10)
        self.sigma_u = np.clip(self.sigma_u, 1e-6, 10)
        self.sigma_v = np.clip(self.sigma_v, 1e-6, 10)
        for user_idx in range(self.n_users):
            user_ratings = [(j, r) for (u, j, r) in train_indices if u == user_idx]
            if user_ratings:
                self._update_user_factor(user_idx, user_ratings)

        for joke_idx in range(self.n_jokes):
            joke_ratings = [(u, r) for (u, j, r) in train_indices if j == joke_idx]
            if joke_ratings:
                self._update_item_factor(joke_idx, joke_ratings)

        self.mu_u = np.clip(self.mu_u, -10, 10)
        self.mu_v = np.clip(self.mu_v, -10, 10)
        self.sigma_u = np.clip(self.sigma_u, 1e-6, 10)
        self.sigma_v = np.clip(self.sigma_v, 1e-6, 10)
        elbo = self._compute_elbo(train_indices)
        return elbo

    def _update_user_factor(self, user_idx, user_ratings):
        joke_indices = [j for j, r in user_ratings]
        ratings = np.array([r for j, r in user_ratings])

        mu_v_relevant = self.mu_v[joke_indices]
        sigma_v_relevant = self.sigma_v[joke_indices]
        precision_prior = 1.0 / self.prior_variance
        precision_likelihood = np.sum(
            (mu_v_relevant**2 + sigma_v_relevant) / self.noise_variance, axis=0
        )
        new_precision = precision_prior + precision_likelihood
        new_precision = np.maximum(new_precision, 1e-10)
        self.sigma_u[user_idx] = 1.0 / new_precision

        # d/d_mu_u E[logp(r|u,v) = (1/sigma_noise^2) *sum_j (r_ij - mu_u^T mu_v_j)*mu_v_j
        residuals = ratings - np.sum(self.mu_u[user_idx] * mu_v_relevant, axis=1)
        gradient = np.sum(residuals[:, np.newaxis] * mu_v_relevant, axis=0) / self.noise_variance
        self.mu_u[user_idx] = (1 - self.learning_rate) * self.mu_u[user_idx] + \
                              self.learning_rate * self.sigma_u[user_idx] * gradient

    def _update_item_factor(self, joke_idx, joke_ratings):
        user_indices = [u for u, r in joke_ratings]
        ratings = np.array([r for u, r in joke_ratings])
        #expected
        mu_u_relevant = self.mu_u[user_indices]
        sigma_u_relevant = self.sigma_u[user_indices]
        precision_prior = 1.0 / self.prior_variance
        precision_likelihood = np.sum(
            (mu_u_relevant**2 + sigma_u_relevant) / self.noise_variance, axis=0
        )
        new_precision = precision_prior + precision_likelihood
        new_precision = np.maximum(new_precision, 1e-10)
        self.sigma_v[joke_idx] = 1.0 / new_precision
        residuals = ratings - np.sum(mu_u_relevant * self.mu_v[joke_idx], axis=1)
        gradient = np.sum(residuals[:, np.newaxis] * mu_u_relevant, axis=0) / self.noise_variance
        self.mu_v[joke_idx] = (1 - self.learning_rate) * self.mu_v[joke_idx] + \
                              self.learning_rate * self.sigma_v[joke_idx] * gradient

    def _compute_elbo(self, train_indices):
        elbo = 0.0
        for user_idx, joke_idx, rating in train_indices:
            pred_mean = np.dot(self.mu_u[user_idx], self.mu_v[joke_idx])
            pred_var = (np.sum(self.sigma_u[user_idx] * self.sigma_v[joke_idx]) +
                       np.sum(self.sigma_u[user_idx] * self.mu_v[joke_idx]**2) +
                       np.sum(self.mu_u[user_idx]**2 * self.sigma_v[joke_idx]) +
                       self.noise_variance)
            pred_var = max(pred_var, 1e-6)
            elbo += -0.5 * np.log(2 * np.pi * pred_var) - 0.5 * (rating - pred_mean)**2 / pred_var

        # KL(q(U) || p(U))= 0.5*sum[log(prior_var/post_var)-1 + post_var/prior_var+ post_mean^2/prior_var]
        kl_u = 0.5 * np.sum(
            np.log(self.prior_variance / self.sigma_u) - 1 +
            self.sigma_u / self.prior_variance +
            self.mu_u**2 / self.prior_variance
        )

        # KL(q(V)||p(V))
        kl_v = 0.5 * np.sum(
            np.log(self.prior_variance / self.sigma_v) - 1 +
            self.sigma_v / self.prior_variance +
            self.mu_v**2 / self.prior_variance
        )

        # ELBO= likelihood-KL divergences
        elbo -= (kl_u + kl_v)
        return elbo

    def predict(self, user_id, joke_id, return_uncertainty=False):
        if user_id not in self.user_to_idx or joke_id not in self.joke_to_idx:
            if return_uncertainty:
                return np.nan, np.nan
            return np.nan

        user_idx = self.user_to_idx[user_id]
        joke_idx = self.joke_to_idx[joke_id]
        pred_mean = np.dot(self.mu_u[user_idx], self.mu_v[joke_idx])
        if return_uncertainty:
            pred_var = (np.sum(self.sigma_u[user_idx] * self.sigma_v[joke_idx]) +
                       np.sum(self.sigma_u[user_idx] * self.mu_v[joke_idx]**2) +
                       np.sum(self.mu_u[user_idx]**2 * self.sigma_v[joke_idx]) +
                       self.noise_variance)

            pred_var = max(pred_var, 1e-6)
            return pred_mean, np.sqrt(pred_var)
        return pred_mean

    def predict_batch(self, test_data, return_uncertainty=False):
        if return_uncertainty:
            predictions = []
            uncertainties = []
            for user_id, joke_id, _ in test_data:
                pred, unc = self.predict(user_id, joke_id, return_uncertainty=True)
                predictions.append(pred)
                uncertainties.append(unc)
            return np.array(predictions), np.array(uncertainties)
        else:
            predictions = []
            for user_id, joke_id, _ in test_data:
                pred = self.predict(user_id, joke_id)
                predictions.append(pred)
            return np.array(predictions)

    def sample_predictions(self, user_id, joke_id, n_samples=100):
        if user_id not in self.user_to_idx or joke_id not in self.joke_to_idx:
            return np.full(n_samples, np.nan)

        user_idx = self.user_to_idx[user_id]
        joke_idx = self.joke_to_idx[joke_id]
        u_samples = np.random.normal(self.mu_u[user_idx],
                                   np.sqrt(self.sigma_u[user_idx]),
                                   (n_samples, self.n_factors))
        v_samples = np.random.normal(self.mu_v[joke_idx],
                                   np.sqrt(self.sigma_v[joke_idx]),
                                   (n_samples, self.n_factors))
        pred_samples = np.sum(u_samples * v_samples, axis=1)
        pred_samples += np.random.normal(0, np.sqrt(self.noise_variance), n_samples)
        return pred_samples

    def evaluate(self, test_data):
        predictions = self.predict_batch(test_data)
        actual = np.array([rating for _, _, rating in test_data])
        valid_mask = ~np.isnan(predictions)
        predictions = predictions[valid_mask]
        actual = actual[valid_mask]
        if len(predictions) == 0:
            return np.inf
        return np.sqrt(mean_squared_error(actual, predictions))

    def get_user_recommendations(self, user_id, n_recommendations=10, return_uncertainty=False):
        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]
        pred_means = self.mu_u[user_idx] @ self.mu_v.T
        if return_uncertainty:
            pred_vars = np.zeros(self.n_jokes)
            for joke_idx in range(self.n_jokes):
                pred_vars[joke_idx] = (np.dot(self.sigma_u[user_idx], self.sigma_v[joke_idx]) +
                                     np.dot(self.sigma_u[user_idx], self.mu_v[joke_idx]**2) +
                                     np.dot(self.mu_u[user_idx]**2, self.sigma_v[joke_idx]) +
                                     self.noise_variance)
            pred_stds = np.sqrt(pred_vars)
            top_jokes = np.argsort(pred_means)[::-1][:n_recommendations]
            recommendations = [(self.idx_to_joke[joke_idx], pred_means[joke_idx], pred_stds[joke_idx])
                             for joke_idx in top_jokes]
            return recommendations
        else:
            top_jokes = np.argsort(pred_means)[::-1][:n_recommendations]
            recommendations = [(self.idx_to_joke[joke_idx], pred_means[joke_idx])
                             for joke_idx in top_jokes]
            return recommendations

def load_jester_data(file_path=False, sample_size=None):
    """Load Jester dataset (same as before)"""
    if file_path is False:
        # Simulate Jester-like data for demonstration
        np.random.seed(42)
        n_users = 1000
        n_jokes = 100

        # Generate some realistic joke rating patterns
        data = []
        for user_id in range(n_users):
            # Each user rates 15-80 jokes
            n_ratings = np.random.randint(15, 81)
            rated_jokes = np.random.choice(n_jokes, n_ratings, replace=False)

            # User preference bias
            user_bias = np.random.normal(0, 2)

            for joke_id in rated_jokes:
                # Joke quality bias
                joke_bias = np.random.normal(0, 1.5)

                # Generate rating with some noise
                rating = user_bias + joke_bias + np.random.normal(0, 1)
                rating = np.clip(rating, -10, 10)  # Jester scale

                data.append((user_id, joke_id, rating))

        if sample_size:
            data = data[:sample_size]

        return data
    else:
        # Real data loading logic (same as before)
        file_paths=["jester-data-1.xls", "jester-data-2.xls", "jester-data-3.xls"]
        all_data = []
        user_id_offset = 0

        for file_path in file_paths:
            print(f"Loading {file_path}...")
            df = pd.read_excel(file_path, header=None)

            if df.iloc[:, 0].max() > 100:
                df = df.iloc[:, 1:]

            for i, row in df.iterrows():
                for joke_id, rating in enumerate(row):
                    if rating != 99:
                        all_data.append((user_id_offset + i, joke_id, float(rating)))

            user_id_offset += len(df)

        if sample_size:
            all_data = all_data[:sample_size]

        return all_data


def run_bayesian_experiment():
    print("Bayesian PMF w/ uncertainty\n")
    print("Loading data...")
    data = load_jester_data(False, sample_size=5000)
    print(f"Loaded {len(data)} ratings")
    print(f"Users: {len(set([x[0] for x in data]))}")
    print(f"Jokes: {len(set([x[1] for x in data]))}")

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print("\nTraining Bayesian PMF...")
    model = BayesianPMF(
        n_factors=10,
        learning_rate=0.1,
        n_iterations=100,
        prior_variance=1.0,
        noise_variance=1.0,
        verbose=True
    )
    model.fit(train_data, val_data)

    print("\nEvaluating model...")
    train_rmse = model.evaluate(train_data)
    val_rmse = model.evaluate(val_data)
    test_rmse = model.evaluate(test_data)
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print("\nAnalyzing prediction uncertainty...")
    sample_test = test_data[:100]
    predictions, uncertainties = model.predict_batch(sample_test, return_uncertainty=True)
    actual = np.array([x[2] for x in sample_test])
    valid_mask = ~np.isnan(predictions)
    predictions = predictions[valid_mask]
    uncertainties = uncertainties[valid_mask]
    actual = actual[valid_mask]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(model.elbo_history, label='ELBO')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('ELBO')
    axes[0, 0].set_title('Evidence Lower Bound')
    axes[0, 0].grid(True)
    axes[0, 1].plot(model.train_losses, label='Training RMSE')
    axes[0, 1].plot(model.val_losses, label='Validation RMSE')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Learning Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Predictions vs actual w/ uncertainty
    axes[1, 0].errorbar(actual, predictions, yerr=uncertainties,
                       fmt='o', alpha=0.6, capsize=2)
    axes[1, 0].plot([-10, 10], [-10, 10], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('Actual Rating')
    axes[1, 0].set_ylabel('Predicted Rating')
    axes[1, 0].set_title('Predictions with Uncertainty')
    axes[1, 0].grid(True)
    # Uncertainty vs absolute error
    abs_errors = np.abs(actual - predictions)
    axes[1, 1].scatter(uncertainties, abs_errors, alpha=0.6)
    axes[1, 1].set_xlabel('Prediction Uncertainty (σ)')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Uncertainty vs Error')
    axes[1, 1].grid(True)
    plt.tight_layout()
    plt.show()

    print("\nSample predictions with uncertainty:")
    for i in range(5):
        user_id, joke_id, actual_rating = sample_test[i]
        pred_mean, pred_std = model.predict(user_id, joke_id, return_uncertainty=True)
        if not np.isnan(pred_mean):
            print(f"User {user_id}, Joke {joke_id}: "
                  f"Actual = {actual_rating:.2f}, "
                  f"Predicted = {pred_mean:.2f} ± {pred_std:.2f}")

    print("\nRecommendations with uncertainty for user 0:")
    recommendations = model.get_user_recommendations(0, n_recommendations=5, return_uncertainty=True)
    for joke_id, mean_score, std_score in recommendations:
        print(f"Joke {joke_id}: {mean_score:.2f} ± {std_score:.2f}")

    print("\nPosterior predictive samples for user 0, joke 1:")
    samples = model.sample_predictions(0, 1, n_samples=10)
    if not np.isnan(samples[0]):
        print(f"Sample predictions: {samples}")
        print(f"Sample mean: {np.mean(samples):.2f}")
        print(f"Sample std: {np.std(samples):.2f}")

    return model

if __name__ == "__main__":
    bayesian_model = run_bayesian_experiment()
    # map_model, bayesian_model = compare_map_vs_bayesian()
