import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ProbabilisticMatrixFactorization:
    """
    Model:
    - R_ij ~ N(U_i^T V_j, sigma_r^2)
    - U_i ~ N(0, sigma_u^2 * I)
    - V_j ~ N(0, sigma_v^2 * I)
    """

    def __init__(self, n_factors=10, learning_rate=0.01, regularization=0.01,
                 n_iterations=100, verbose=True):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_iterations = n_iterations
        self.verbose = verbose

    def fit(self, train_data, val_data=None):
        users = set([x[0] for x in train_data])
        jokes = set([x[1] for x in train_data])
        self.n_users = len(users)
        self.n_jokes = len(jokes)
        self.user_to_idx = {user: idx for idx, user in enumerate(sorted(users))}
        self.joke_to_idx = {joke: idx for idx, joke in enumerate(sorted(jokes))}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_joke = {idx: joke for joke, idx in self.joke_to_idx.items()}
        self.U = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.V = np.random.normal(0, 0.1, (self.n_jokes, self.n_factors))

        train_indices = [(self.user_to_idx[u], self.joke_to_idx[j], r)
                        for u, j, r in train_data]

        self.train_losses = []
        self.val_losses = []
        # training loop
        for iteration in range(self.n_iterations):
            np.random.shuffle(train_indices)
            train_loss = 0
            for user_idx, joke_idx, rating in train_indices:
                # predict
                pred = np.dot(self.U[user_idx], self.V[joke_idx])
                error = rating - pred
                u_old = self.U[user_idx].copy()
                v_old = self.V[joke_idx].copy()
                self.U[user_idx] += self.learning_rate * (
                    error * v_old - self.regularization * u_old
                )
                self.V[joke_idx] += self.learning_rate * (
                    error * u_old - self.regularization * v_old
                )
                train_loss += error ** 2

            train_loss = np.sqrt(train_loss / len(train_indices))
            self.train_losses.append(train_loss)
            # validation loss
            if val_data:
                val_loss = self.evaluate(val_data)
                self.val_losses.append(val_loss)
                if self.verbose and iteration % 10 == 0:
                    print(f"Iteration {iteration}: Train RMSE = {train_loss:.4f}, "
                          f"Val RMSE = {val_loss:.4f}")
            else:
                if self.verbose and iteration % 10 == 0:
                    print(f"Iteration {iteration}: Train RMSE = {train_loss:.4f}")

    def predict(self, user_id, joke_id):
        if user_id not in self.user_to_idx or joke_id not in self.joke_to_idx:
            return np.nan

        user_idx = self.user_to_idx[user_id]
        joke_idx = self.joke_to_idx[joke_id]
        return np.dot(self.U[user_idx], self.V[joke_idx])

    def predict_batch(self, test_data):
        predictions = []
        for user_id, joke_id, _ in test_data:
            pred = self.predict(user_id, joke_id)
            predictions.append(pred)
        return np.array(predictions)

    def evaluate(self, test_data):
        predictions = self.predict_batch(test_data)
        actual = np.array([rating for _, _, rating in test_data])
        # remove irrelevant NaN predictions not in training
        valid_mask = ~np.isnan(predictions)
        predictions = predictions[valid_mask]
        actual = actual[valid_mask]
        if len(predictions) == 0:
            return np.inf

        return np.sqrt(mean_squared_error(actual, predictions))

    def get_user_recommendations(self, user_id, n_recommendations=10):
        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]
        scores = self.U[user_idx] @ self.V.T
        top_jokes = np.argsort(scores)[::-1][:n_recommendations]
        recommendations = [(self.idx_to_joke[joke_idx], scores[joke_idx]) for joke_idx in top_jokes]
        return recommendations

def load_jester_data(sample_size=None):
    file_paths=["jester-data-1.xls", "jester-data-2.xls", "jester-data-3.xls"]
    all_data = []
    user_id_offset = 0  # Tracking global user IDs across files

    for file_path in file_paths:
        print(f"Loading {file_path}...")
        df = pd.read_excel(file_path, header=None)
        if df.iloc[:, 0].max() > 100:
            df = df.iloc[:, 1:]
        for i, row in df.iterrows():
            for joke_id, rating in enumerate(row):
                if rating != 99:  # 99 indicates missing rating
                    all_data.append((user_id_offset + i, joke_id, float(rating)))
        user_id_offset += len(df)

    if sample_size:
        all_data = all_data[:sample_size]
    return all_data

def run_experiment():
    print("Loading Jester data...")
    data = load_jester_data(sample_size=10000)
    print(f"Loaded {len(data)} ratings")
    print(f"Users: {len(set([x[0] for x in data]))}")
    print(f"Jokes: {len(set([x[1] for x in data]))}")

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    print("\nTraining PMF model...")
    model = ProbabilisticMatrixFactorization(
        n_factors=10,
        learning_rate=0.01,
        regularization=0.01,
        n_iterations=100,
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

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(model.train_losses, label='Training RMSE')
    plt.plot(model.val_losses, label='Validation RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    sample_test = test_data[:200]
    predictions = model.predict_batch(sample_test)
    actual = [x[2] for x in sample_test]
    valid_mask = ~np.isnan(predictions)
    predictions = predictions[valid_mask]
    actual = np.array(actual)[valid_mask]
    plt.scatter(actual, predictions, alpha=0.6)
    plt.plot([-10, 10], [-10, 10], 'r--', alpha=0.8)
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.title('Predictions vs Actual')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nSample recommendations for user 0:")
    recommendations = model.get_user_recommendations(0, n_recommendations=5)
    for joke_id, score in recommendations:
        print(f"Joke {joke_id}: {score:.2f}")

    return model, train_data, val_data, test_data

class ExperimentRunner:
    def __init__(self, data):
        self.data = data
        self.results = {}

    def factor_analysis(self, factor_range=[5, 10, 15, 20, 30]):
        print("Running factor analysis...")
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        results = {'factors': [], 'train_rmse': [], 'val_rmse': [], 'test_rmse': []}
        for n_factors in factor_range:
            print(f"Testing {n_factors} factors...")
            model = ProbabilisticMatrixFactorization(
                n_factors=n_factors,
                learning_rate=0.01,
                regularization=0.01,
                n_iterations=50,
                verbose=False
            )
            model.fit(train_data, val_data)
            train_rmse = model.evaluate(train_data)
            val_rmse = model.evaluate(val_data)
            test_rmse = model.evaluate(test_data)
            results['factors'].append(n_factors)
            results['train_rmse'].append(train_rmse)
            results['val_rmse'].append(val_rmse)
            results['test_rmse'].append(test_rmse)
            print(f"  RMSE - Train: {train_rmse:.4f}, Val: {val_rmse:.4f}, Test: {test_rmse:.4f}")

        self.results['factor_analysis'] = results
        return results

    def learning_curve_analysis(self, data_fractions=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0]):
        print("Running learning curve analysis...")
        results = {'data_fraction': [], 'train_rmse': [], 'val_rmse': [], 'test_rmse': []}
        train_val_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        for fraction in data_fractions:
            print(f"Testing with {fraction*100:.0f}% of training data...")
            sample_size = int(len(train_val_data) * fraction)
            sampled_data = train_val_data[:sample_size]
            train_data, val_data = train_test_split(sampled_data, test_size=0.2, random_state=42)
            model = ProbabilisticMatrixFactorization(
                n_factors=10,
                learning_rate=0.01,
                regularization=0.01,
                n_iterations=50,
                verbose=False
            )
            model.fit(train_data, val_data)
            train_rmse = model.evaluate(train_data)
            val_rmse = model.evaluate(val_data)
            test_rmse = model.evaluate(test_data)
            results['data_fraction'].append(fraction)
            results['train_rmse'].append(train_rmse)
            results['val_rmse'].append(val_rmse)
            results['test_rmse'].append(test_rmse)
            print(f"  RMSE - Train: {train_rmse:.4f}, Val: {val_rmse:.4f}, Test: {test_rmse:.4f}")

        self.results['learning_curve'] = results
        return results

    def regularization_analysis(self, reg_values=[0.001, 0.01, 0.1, 0.5, 1.0]):
        print("Running regularization analysis...")
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        results = {'regularization': [], 'train_rmse': [], 'val_rmse': [], 'test_rmse': []}
        for reg in reg_values:
            print(f"Testing regularization {reg}...")
            model = ProbabilisticMatrixFactorization(
                n_factors=10,
                learning_rate=0.01,
                regularization=reg,
                n_iterations=50,
                verbose=False
            )
            model.fit(train_data, val_data)
            train_rmse = model.evaluate(train_data)
            val_rmse = model.evaluate(val_data)
            test_rmse = model.evaluate(test_data)
            results['regularization'].append(reg)
            results['train_rmse'].append(train_rmse)
            results['val_rmse'].append(val_rmse)
            results['test_rmse'].append(test_rmse)
            print(f"  RMSE - Train: {train_rmse:.4f}, Val: {val_rmse:.4f}, Test: {test_rmse:.4f}")

        self.results['regularization_analysis'] = results
        return results

    def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        if 'factor_analysis' in self.results:
            results = self.results['factor_analysis']
            axes[0, 0].plot(results['factors'], results['train_rmse'], 'o-', label='Train')
            axes[0, 0].plot(results['factors'], results['val_rmse'], 's-', label='Validation')
            axes[0, 0].plot(results['factors'], results['test_rmse'], '^-', label='Test')
            axes[0, 0].set_xlabel('Number of Factors')
            axes[0, 0].set_ylabel('RMSE')
            axes[0, 0].set_title('Effect of Number of Latent Factors')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        if 'learning_curve' in self.results:
            results = self.results['learning_curve']
            axes[0, 1].plot(results['data_fraction'], results['train_rmse'], 'o-', label='Train')
            axes[0, 1].plot(results['data_fraction'], results['val_rmse'], 's-', label='Validation')
            axes[0, 1].plot(results['data_fraction'], results['test_rmse'], '^-', label='Test')
            axes[0, 1].set_xlabel('Fraction of Training Data')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].set_title('Learning Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        if 'regularization_analysis' in self.results:
            results = self.results['regularization_analysis']
            axes[1, 0].semilogx(results['regularization'], results['train_rmse'], 'o-', label='Train')
            axes[1, 0].semilogx(results['regularization'], results['val_rmse'], 's-', label='Validation')
            axes[1, 0].semilogx(results['regularization'], results['test_rmse'], '^-', label='Test')
            axes[1, 0].set_xlabel('Regularization Parameter')
            axes[1, 0].set_ylabel('RMSE')
            axes[1, 0].set_title('Effect of Regularization')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        fig.delaxes(axes[1, 1])
        plt.tight_layout()
        plt.show()

def run_full_analysis():
    print("Jester Probabilistic Matrix Factorization Analysis\n")
    # load data
    data = load_jester_data(sample_size=5000)

    # basic experiment
    print("Basic model training:")
    model, train_data, val_data, test_data = run_experiment()

    # systematic analysis
    print("\nSystematic analysis:")
    experiment_runner = ExperimentRunner(data)

    # all analyses
    experiment_runner.factor_analysis()
    experiment_runner.learning_curve_analysis()
    experiment_runner.regularization_analysis()

    experiment_runner.plot_results()
    return experiment_runner

if __name__ == "__main__":
    experiment_runner = run_full_analysis()
