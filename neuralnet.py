import multiprocessing
import os

from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import neural_network
import matplotlib.pyplot as plt
import numpy as np


# GLOBAL VARIABLES
RANDOM_SEED = 14
CORE_COUNT_PERCENTAGE = .75  # NOTE: Any increase past this and the comp is unusable
ONE_HOT_ENCODING = True


class NNClass:
    """ Class for running neural network """

    def __init__(self, name, data, run_config, outdir):
        # Base Problem variables
        self.name = name
        self.title = "Neural Network"
        self.dataset = data
        self.instance = None
        self.config = run_config
        self.out_dir = outdir
        self.opt_list = dict()
        self.curves = dict()
        self.random_seed = RANDOM_SEED
        self.core_count = round(multiprocessing.cpu_count() * CORE_COUNT_PERCENTAGE)
        # For LCA
        self.train_sizes = None
        self.train_scores = None
        self.test_scores = None
        self.fit_times = None
        self.score_times = None
        self.acc_train = None
        self.acc_test = None
        self.loss = None

    def process_instance(self):
        """ Create the sklearn instance for NN """
        self.instance = neural_network.MLPClassifier(hidden_layer_sizes=self.config["hidden_layers_sizes"], activation=self.config["activation"],
                                                     alpha=self.config["alpha"], batch_size=self.config["batch_size"],
                                                     learning_rate=self.config["learning_rate"], learning_rate_init=self.config["learning_rate_init"],
                                                     max_iter=self.config["max_iter"], tol=self.config["tol"], random_state=RANDOM_SEED)

    def run(self):
        """ Run the neural net """
        print(f"Running {self.title}")
        # Generate instance
        self.process_instance()

        # Run fit and predict
        self.instance.fit(self.dataset.x_train, self.dataset.y_train)
        self.dataset.y_predict = self.instance.predict(self.dataset.x_test)

        # Generate scores
        self.acc_train = accuracy_score(self.dataset.y_train, self.instance.predict(self.dataset.x_train))
        self.acc_test = accuracy_score(self.dataset.y_test, self.instance.predict(self.dataset.x_test))
        self.loss = self.instance.loss

    def plot_lca(self):
        """ Plots LCA curve """
        """ NOTE: DO NOT RUN INSIDE POOL """
        self.train_sizes, self.train_scores, self.test_scores, self.fit_times, self.score_times = model_selection.learning_curve(
            self.instance, self.dataset.x_train, self.dataset.y_train, n_jobs=self.core_count, return_times=True)

        # Generate figure
        fig, axes = plt.subplots()
        # Compute stats
        train_scores_mean = np.mean(self.train_scores, axis=1)
        train_scores_std = np.std(self.train_scores, axis=1)
        test_scores_mean = np.mean(self.test_scores, axis=1)
        test_scores_std = np.std(self.test_scores, axis=1)
        fit_times_mean = np.mean(self.fit_times, axis=1)
        fit_times_std = np.std(self.fit_times, axis=1)

        # Plot learning curve
        axes.grid()
        axes.fill_between(
            self.train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes.fill_between(
            self.train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        axes.plot(self.train_sizes, train_scores_mean, "o-", color="r", label="Training score")
        axes.plot(self.train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
        axes.legend(loc="best")
        axes.set_title(f"Learning Curve for Neural Network")
        axes.set_xlabel("Training examples")
        axes.set_ylabel("Accuracy")
        axes.grid(True)

        fig.savefig(os.path.join(self.out_dir, f"{self.name}_lc_curve.png"))

        # Generate figure
        fig, axes = plt.subplots()

        # Plot n_samples vs fit_times
        axes.grid()
        axes.plot(self.train_sizes, fit_times_mean, "o-")
        axes.fill_between(
            self.train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1
        )
        axes.set_xlabel("Training examples")
        axes.set_ylabel("fit_times")
        axes.set_title("Scalability of the model")

        fig.savefig(os.path.join(self.out_dir, f"{self.name}_time_curve.png"))

        # Generate figure
        fig, axes = plt.subplots()

        # Plot fit_time vs score
        axes.grid()
        axes.plot(fit_times_mean, test_scores_mean, "o-")
        axes.fill_between(
            fit_times_mean,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
        )
        axes.set_xlabel("fit_times")
        axes.set_ylabel("Score")
        axes.set_title("Performance of the model")

        fig.savefig(os.path.join(self.out_dir, f"{self.name}_stime_curve.png"))

    def plot_loss(self):
        """ Plot loss curve from NN """
        fig, axes = plt.subplots()

        plt.plot(self.instance.loss_curve_)
        plt.title(f"Loss Curve for Neural Networks")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        fig.savefig(os.path.join(self.out_dir, f"{self.name}_loss_curve.png"))
