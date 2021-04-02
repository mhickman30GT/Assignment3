import multiprocessing
import os
import contextlib
import io
import time
import copy

from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn import cluster
from sklearn import decomposition
from sklearn import random_projection
from sklearn import ensemble
from sklearn import tree
from sklearn.mixture import GaussianMixture
from yellowbrick import cluster as ybcluster
from yellowbrick import text as ybt
from yellowbrick import features

import neuralnet as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# GLOBAL VARIABLES
RANDOM_SEED = 14
CORE_COUNT_PERCENTAGE = .75  # NOTE: Any increase past this and the comp is unusable
ONE_HOT_ENCODING = True


class Pool:
    """Class to run a pool of algorithms"""

    def __init__(self, algorithms, num_cores=4):
        """Constructor for Pool"""
        self.algorithms = algorithms
        self.num_cores = min(num_cores, len(algorithms))

    @staticmethod
    def _run_algorithm(algorithm):
        """Run algorithm (capture and store stdout)"""
        print(f"    Start : {algorithm.name}")
        string_io = io.StringIO()
        with contextlib.redirect_stdout(string_io):
            algorithm.run()
        algorithm.stdout = string_io.getvalue()
        print(f"    Stop  : {algorithm.name}")
        return algorithm

    def run(self):
        """Run algorithms in multiprocessing pool"""
        print(f"Running {len(self.algorithms)} algorithms with {self.num_cores} cores")
        pool = multiprocessing.Pool(processes=self.num_cores)
        self.algorithms = pool.map(self._run_algorithm, self.algorithms)
        pool.close()
        pool.join()


class DataSet:
    """ Class holding values for dataset """

    def __init__(self, name, label, file):
        """ Constructor for Dataset """
        self.data_name = f'{name} Data Set'
        self.file = file
        self.csv = pd.read_csv(self.file)
        self.label = label
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series()
        self.y_test = pd.Series()
        self.y_predict = pd.Series()
        self.x_cluster = pd.DataFrame()
        self.features = list()

    def process(self):
        """ Processes data set """
        # Separate classification labels
        x = self.csv.drop(self.label, 1)
        y = self.csv[self.label]

        # Default to one hot for all sets
        if ONE_HOT_ENCODING:
            x = pd.get_dummies(x, columns=x.select_dtypes(include=[object]).columns)

        self.features = list(x.columns)

        # Split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(
            x, y, stratify=y, test_size=0.25, random_state=RANDOM_SEED)

        # Scale data using standardizer
        standardize = preprocessing.StandardScaler()
        self.x_train = standardize.fit_transform(self.x_train)
        self.x_test = standardize.transform(self.x_test)

    def append_clusters(self, clusters):
        """ Appends clustering data to training/test data """
        df = pd.DataFrame(self.x_train)
        df["Cluster"] = clusters
        self.x_cluster = df


class KMeans:
    """ Class to run K-Means Clustering """

    def __init__(self, name, data, outdir, booster, clusters=8, init=10, max_iter=300, tol=0.0001):
        """ Constructor for K-Means """
        self.name = name
        self.title = "K-Means Clustering"
        self.dataset = data
        self.outdir = outdir
        self.num_clusters = clusters
        self.x_transform = None
        self.labels = None
        self.sil_score = None
        self.score = None
        self.cv_score = None
        self.fit_time = None
        self.learner = booster
        self.instance = cluster.KMeans(n_clusters=clusters, n_init=init,
                                       max_iter=max_iter, tol=tol, random_state=RANDOM_SEED)

    def run(self):
        """ Run the K-Means clustering """
        # Run clustering
        start_time = time.time()
        self.labels = self.instance.fit_predict(self.dataset.x_train)
        self.x_transform = self.instance.fit_transform(self.dataset.x_train)
        self.fit_time = round(time.time() - start_time, 2)

        # Score clustering
        self.score = self.instance.score(self.dataset.x_train)
        self.sil_score = metrics.silhouette_score(self.dataset.x_train, self.labels, random_state=RANDOM_SEED)

        # CV Score via Boosting Learner with cluster data
        self.dataset.append_clusters(self.labels)
        self.learner.fit(self.dataset.x_cluster, self.dataset.y_train)
        self.cv_score = np.mean(
            model_selection.cross_val_score(self.learner, self.dataset.x_cluster, self.dataset.y_train, n_jobs=6,
                                            cv=10))

    def plot(self):
        """ Create the K-Means clustering plots """
        # Create Intercluster Distance plot
        fig, ax = plt.subplots()
        visualizer = ybcluster.InterclusterDistance(self.instance, ax=ax)
        visualizer.fit(self.dataset.x_train)
        visualizer.finalize()
        fig.savefig(os.path.join(self.outdir, f"{self.name}_cluster_distance.png"))

        # Create Silhouette plot
        fig, ax = plt.subplots()
        visualizer = ybcluster.SilhouetteVisualizer(self.instance, ax=ax)
        visualizer.fit(self.dataset.x_train)
        visualizer.finalize()
        fig.savefig(os.path.join(self.outdir, f"{self.name}_silhouette_plot.png"))

        # Create the TSNE plot
        fig, ax = plt.subplots()
        tsne = ybt.TSNEVisualizer(decompose_by=self.num_clusters - 1, ax=ax, random_state=RANDOM_SEED)
        tsne.fit(self.x_transform, self.labels)
        tsne.finalize()
        fig.savefig(os.path.join(self.outdir, f"{self.name}_tsne.png"))


class EMCluster:
    """ Class to run EM Clustering """

    def __init__(self, name, data, outdir, booster, components=1, init=1, max_iter=100, tol=1e-3):
        """ Constructor for EM """
        self.name = name
        self.title = "EM Clustering"
        self.dataset = data
        self.outdir = outdir
        self.learner = booster
        self.num_components = components
        self.x_transform = None
        self.labels = None
        self.score = None
        self.cv_score = None
        self.mutual_info = None
        self.aic = None
        self.bic = None
        self.fit_time = None
        self.instance = GaussianMixture(n_components=components, n_init=init,
                                        max_iter=max_iter, tol=tol, random_state=RANDOM_SEED)

    def run(self):
        """ Run the EM clustering """
        # Run clustering
        start_time = time.time()
        self.labels = self.instance.fit_predict(self.dataset.x_train)
        # self.x_transform, self.sample_labels = self.instance.sample(self.num_components)
        self.fit_time = round(time.time() - start_time, 2)

        # Score clustering
        self.score = self.instance.score(self.dataset.x_train)
        self.mutual_info = metrics.adjusted_mutual_info_score(self.dataset.y_train, self.labels)
        self.aic = self.instance.aic(self.dataset.x_train)
        self.bic = self.instance.bic(self.dataset.x_train)

        # CV Score via Boosting Learner with cluster data
        self.dataset.append_clusters(self.labels)
        self.learner.fit(self.dataset.x_cluster, self.dataset.y_train)
        self.cv_score = np.mean(
            model_selection.cross_val_score(self.learner, self.dataset.x_cluster, self.dataset.y_train, n_jobs=6,
                                            cv=10))

    def plot(self):
        """ Create the EM clustering plots """
        # Create the TSNE plot
        fig, ax = plt.subplots()
        tsne = ybt.TSNEVisualizer(decompose_by=self.dataset.x_train.shape[1] - 1, ax=ax, random_state=RANDOM_SEED)
        tsne.fit(self.dataset.x_train, ["c{}".format(c) for c in self.labels])
        tsne.finalize()
        fig.savefig(os.path.join(self.outdir, f"{self.name}_tsne.png"))


class PCA:
    """ Class to run PCA Dimensionality Reduction """

    def __init__(self, name, data, outdir, boost, components=1, tol=1e-3):
        """ Constructor for PCA """
        self.name = name
        self.title = "PCA"
        self.dataset = data
        self.outdir = outdir
        self.num_components = components
        self.x_transform = None
        self.labels = None
        self.score = None
        self.eigen_values = None
        self.eigen_score = None
        self.cv_score = None
        self.fit_time = None
        self.boost_time = None
        self.reconstruction = None
        self.instance = decomposition.PCA(n_components=components, tol=tol, random_state=RANDOM_SEED)
        self.learner = boost

    def run(self):
        """ Run PCA """
        # Run PCA
        start_time = time.time()
        self.x_transform = self.instance.fit_transform(self.dataset.x_train)
        self.fit_time = round(time.time() - start_time, 2)

        # Score PCA and eigen values
        self.score = self.instance.score(self.dataset.x_train)
        self.eigen_values = self.instance.explained_variance_
        self.eigen_score = self.instance.explained_variance_ratio_.sum()

        # Calculate reconstruction error
        inverse_data = np.linalg.pinv(self.instance.components_.T)
        reconstruction_data = self.x_transform.dot(inverse_data)
        self.reconstruction = np.sqrt(metrics.mean_squared_error(self.dataset.x_train, reconstruction_data))
        self.reconstruction /= np.sqrt(np.mean(self.dataset.x_train ** 2))

        # CV Score via Boosting Learner with reduced data
        start_time = time.time()
        self.learner.fit(self.x_transform, self.dataset.y_train)
        self.cv_score = np.mean(
            model_selection.cross_val_score(self.learner, self.x_transform, self.dataset.y_train, n_jobs=6, cv=10))
        self.boost_time = round(time.time() - start_time, 2)

    def plot(self):
        """ Create PCA plots """
        # Create eigen value plot
        fig, ax = plt.subplots()
        x_data = list(range(1, len(self.eigen_values) + 1))
        y_data = self.eigen_values
        ax.plot(x_data, y_data, marker='o', linestyle='None', markersize=10)
        ax.set_title(f"Variance over Components")
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance")
        fig.savefig(os.path.join(self.outdir, f"{self.name}_eigenvalues.png"))


class ICA:
    """ Class to run ICA Dimensionality Reduction """

    def __init__(self, name, data, outdir, boost, components=1, iter=100, tol=1e-4):
        """ Constructor for ICA """
        self.name = name
        self.title = "ICA"
        self.dataset = data
        self.outdir = outdir
        self.num_components = components
        self.x_transform = None
        self.labels = None
        self.kurtosis = None
        self.kurtosis_data = None
        self.fit_time = None
        self.learner = boost
        self.cv_score = None
        self.boost_time = None
        self.reconstruction = None
        self.instance = decomposition.FastICA(n_components=components, max_iter=iter,
                                              tol=tol, random_state=RANDOM_SEED)

    def run(self):
        """ Run ICA """
        # Run ICA
        start_time = time.time()
        self.x_transform = self.instance.fit_transform(self.dataset.x_train)
        self.fit_time = round(time.time() - start_time, 2)

        # Access kurtosis
        self.kurtosis_data = pd.DataFrame(self.x_transform).kurt(axis=0)
        self.kurtosis = self.kurtosis_data.abs().sum()

        # Calculate reconstruction error
        inverse_data = np.linalg.pinv(self.instance.components_.T)
        reconstruction_data = self.x_transform.dot(inverse_data)
        self.reconstruction = np.sqrt(metrics.mean_squared_error(self.dataset.x_train, reconstruction_data))
        self.reconstruction /= np.sqrt(np.mean(self.dataset.x_train ** 2))

        # CV Score via Boosting Learner with reduced data
        start_time = time.time()
        self.learner.fit(self.x_transform, self.dataset.y_train)
        self.cv_score = np.mean(
            model_selection.cross_val_score(self.learner, self.x_transform, self.dataset.y_train, n_jobs=6, cv=10))
        self.boost_time = round(time.time() - start_time, 2)

    def plot(self):
        """ Create ICA plots """
        # Create kurtosis plot
        fig, ax = plt.subplots()
        x_data = list(range(1, len(self.kurtosis_data) + 1))
        y_data = self.kurtosis_data
        ax.plot(x_data, y_data, marker='o', linestyle='None', markersize=10)
        ax.set_title(f"Kurtosis over Components")
        ax.set_xlabel("Component")
        ax.set_ylabel("Kurtosis")
        fig.savefig(os.path.join(self.outdir, f"{self.name}_kurtosis.png"))


class RP:
    """ Class to run RP Dimensionality Reduction """

    def __init__(self, name, data, outdir, boost, components='auto', eps=0.1):
        """ Constructor for RP """
        self.name = name
        self.title = "Random Projection"
        self.dataset = data
        self.outdir = outdir
        self.num_components = components
        self.x_transform = None
        self.labels = None
        self.reconstruction = None
        self.fit_time = None
        self.learner = boost
        self.cv_score = None
        self.boost_time = None
        self.instance = random_projection.GaussianRandomProjection(n_components=components, eps=eps,
                                                                   random_state=RANDOM_SEED)

    def run(self):
        """ Run RP """
        # Run random projection
        start_time = time.time()
        self.x_transform = self.instance.fit_transform(self.dataset.x_train)
        self.fit_time = round(time.time() - start_time, 2)

        # Calculate reconstruction error
        inverse_data = np.linalg.pinv(self.instance.components_.T)
        reconstruction_data = self.x_transform.dot(inverse_data)
        self.reconstruction = np.sqrt(metrics.mean_squared_error(self.dataset.x_train, reconstruction_data))
        self.reconstruction /= np.sqrt(np.mean(self.dataset.x_train ** 2))

        # CV Score via Boosting Learner with reduced data
        start_time = time.time()
        self.learner.fit(self.x_transform, self.dataset.y_train)
        self.cv_score = np.mean(
            model_selection.cross_val_score(self.learner, self.x_transform, self.dataset.y_train, n_jobs=6, cv=10))
        self.boost_time = round(time.time() - start_time, 2)

    def plot(self):
        """ Create RP plots """
        print("No plots for RP")


class KPCA:
    """ Class to run KernalPCA Dimensionality Reduction """

    def __init__(self, name, data, outdir, boost, kernal_config, components=1):
        """ Constructor for KernalPCA """
        self.name = name
        self.title = "Kernal PCA"
        self.dataset = data
        self.outdir = outdir
        self.num_components = components
        self.x_transform = None
        self.labels = None
        self.eigen_values = None
        self.eigen_score = None
        self.cv_score = None
        self.fit_time = None
        self.boost_time = None
        self.learner = boost
        self.reconstruction = None
        self.instance = decomposition.KernelPCA(n_components=components, kernel=kernal_config["kernel"],
                                                degree=kernal_config["degree"], coef0=kernal_config["coef0"],
                                                alpha=kernal_config["alpha"], random_state=RANDOM_SEED)

    def run(self):
        """ Run Kernel PCA """
        # Run KPCA
        start_time = time.time()
        self.x_transform = self.instance.fit_transform(self.dataset.x_train)
        self.fit_time = round(time.time() - start_time, 2)

        # Score KPCA and eigen values
        self.eigen_values = self.x_transform.var(axis=0)
        self.eigen_score = self.eigen_values.sum()

        # Calculate reconstruction error
        #inverse_data = np.linalg.pinv(self.instance.components_.T)
        #reconstruction_data = self.x_transform.dot(inverse_data)
        #self.reconstruction = np.sqrt(metrics.mean_squared_error(self.dataset.x_train, reconstruction_data))
        #self.reconstruction /= np.sqrt(np.mean(self.dataset.x_train ** 2))

        # CV Score via Boosting Learner with reduced data
        start_time = time.time()
        self.learner.fit(self.x_transform, self.dataset.y_train)
        self.cv_score = np.mean(
            model_selection.cross_val_score(self.learner, self.x_transform, self.dataset.y_train, n_jobs=6, cv=10))
        self.boost_time = round(time.time() - start_time, 2)

    def plot(self):
        """ Create Kernel PCA plots """
        # Create eigen value plot
        fig, ax = plt.subplots()
        x_data = list(range(1, len(self.eigen_values) + 1))
        y_data = self.eigen_values
        ax.plot(x_data, y_data, marker='o', linestyle='None', markersize=10)
        ax.set_title(f"Variance over Components")
        ax.set_xlabel("Component")
        ax.set_ylabel("Variance")
        fig.savefig(os.path.join(self.outdir, f"{self.name}_eigenvalues.png"))


class ExperimentClass:
    """ Class for running experiments """

    def __init__(self, name, data, part, rtype,
                 clust, reducer, run_config, outdir):
        # Base Problem variables
        self.name = name
        self.data_class = data
        self.part = part
        self.config = run_config
        self.out_dir = outdir
        self.random_seed = RANDOM_SEED
        self.core_count = round(multiprocessing.cpu_count() * CORE_COUNT_PERCENTAGE)
        self.type = rtype
        self.cluster = clust
        self.reduction = reducer
        self.cluster_class = None
        self.reduction_class = None
        self.neural_class = None
        self.boost = None

    def process_instances(self):
        """ Process instances for unsupervised learners """
        print(f"Processing inputs for part {self.part} {self.type}")

        # Number workaround for generation
        if self.cluster:
            config = self.config[self.cluster]
            if self.cluster == "kmeans":
                number = config["n_clusters"]
            else:
                number = config["components"]
            # Generate instance
            self.cluster_class = self.generate_cluster(self.cluster, self.data_class, number, self.config[self.cluster])
        # Generate reducer
        if self.reduction:
            self.reduction_class = self.generate_reducer(self.reduction, self.data_class,
                                                         self.config[self.reduction]["components"],
                                                         self.config[self.reduction])



    def generate_cluster(self, experiment, data, number, config):
        """ Generate clustering classes """
        if experiment == "kmeans":
            return KMeans(self.name, data, self.out_dir, self.boost, number,
                          config["init"], config["max_iter"], config["tol"])
        else:
            return EMCluster(self.name, data, self.out_dir, self.boost, number,
                             config["init"], config["max_iter"], config["tol"])

    def generate_reducer(self, experiment, data, n, config):
        """ Generate dimensionality reducer classes """
        if experiment == "pca":
            return PCA(self.name, data, self.out_dir,
                       self.boost, n, config["tol"])
        elif experiment == "ica":
            return ICA(self.name, data, self.out_dir,
                       self.boost, n, config["iter"], config["tol"])
        elif experiment == "rp":
            return RP(self.name, data, self.out_dir,
                      self.boost, n, config["eps"])
        else:
            return KPCA(self.name, data, self.out_dir, self.boost, config, components=n)

    def plot_data(self):
        """ Plot data analysis """

        # Create Normalized Distribution plot
        fig, ax = plt.subplots()
        visualizer = features.Rank1D(ax=ax, features=self.data_class.features)
        visualizer.fit(self.data_class.x_train, self.data_class.y_train)
        visualizer.transform(self.data_class.x_train)
        visualizer.finalize()
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_normaldistribute_plot.png"))

        # Create Colinear plot
        fig, ax = plt.subplots()
        visualizer = features.Rank2D(ax=ax, algorithm='pearson', features=self.data_class.features)
        visualizer.fit(self.data_class.x_train, self.data_class.y_train)
        visualizer.transform(self.data_class.x_train)
        visualizer.finalize()
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_colinear_plot.png"))

        # Create Covariance plot
        fig, ax = plt.subplots()
        visualizer = features.Rank2D(ax=ax, algorithm='covariance', features=self.data_class.features)
        visualizer.fit(self.data_class.x_train, self.data_class.y_train)
        visualizer.transform(self.data_class.x_train)
        visualizer.finalize()
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_covariance_plot.png"))

    def plot_kelbow(self, kmin, kmax):
        """ Create elbow plot for clusters """
        # Generate original instance
        instance = self.generate_cluster(self.cluster, self.data_class, self.config[self.cluster]["n_clusters"],
                                         self.config[self.cluster])

        # Create Elbow Method
        fig, ax = plt.subplots()
        visualizer = ybcluster.KElbowVisualizer(instance.instance, k=(kmin, kmax), ax=ax)
        visualizer.fit(instance.dataset.x_train)
        visualizer.finalize()
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_KElbow_plot.png"))

    def km_tuning(self, nmin, nmax):
        """ Tune n_components for KMeans clustering """
        # Init range and instances
        nrange = range(nmin, nmax)
        inst_list = dict()
        cv_df = pd.DataFrame(index=nrange, columns=['CV'])

        # Generate data
        for n in nrange:
            l_inst = self.generate_cluster(self.cluster, self.data_class, n, self.config[self.cluster])
            l_inst.run()
            cv_df.at[n] = l_inst.cv_score
            inst_list[n] = l_inst

        # Create score plot
        fig, ax = plt.subplots()
        ax.set_title("Cross Val over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("CV")
        cv_df.reset_index().plot(kind='line', ax=ax, label="CV",
                                 x='index', y='CV')
        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_cross_validation.png"))

    def em_tuning(self, nmin, nmax):
        """ Tune n_components for EM clustering """
        # Init range and instances
        nrange = range(nmin, nmax)
        inst_list = dict()
        tune_df = pd.DataFrame(index=nrange, columns=['AIC', 'BIC'])
        cv_df = pd.DataFrame(index=nrange, columns=['CV'])

        # Generate data
        for n in nrange:
            l_inst = self.generate_cluster(self.cluster, self.data_class, n, self.config[self.cluster])
            l_inst.run()
            tune_df.at[n] = [l_inst.aic, l_inst.bic]
            cv_df.at[n] = l_inst.cv_score
            inst_list[n] = l_inst

        # Create plots
        fig, ax = plt.subplots()
        ax.set_title("Scores over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("Score")
        tune_df.reset_index().plot(kind='line', ax=ax, label="AIC",
                                   x='index', y='AIC')
        tune_df.reset_index().plot(kind='line', ax=ax, label="BIC",
                                   x='index', y='BIC', color='red')
        ax.legend(loc="best")

        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_em_tuning.png"))

        # Create score plot
        fig, ax = plt.subplots()
        ax.set_title("Cross Val over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("CV")
        cv_df.reset_index().plot(kind='line', ax=ax, label="CV",
                                 x='index', y='CV')
        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_cross_validation.png"))

    def pca_tuning(self, nmin, nmax):
        """ Tune n_components for PCA """
        # Init range and instances
        nrange = range(nmin, nmax)
        inst_list = dict()
        tune_df = pd.DataFrame(index=nrange, columns=['Variance Ratio'])
        cv_df = pd.DataFrame(index=nrange, columns=['CV', 'Time'])
        re_df = pd.DataFrame(index=nrange, columns=['Reconstruction Error'])

        # Generate data
        for n in nrange:
            l_inst = self.generate_reducer(self.reduction, self.data_class, n, self.config[self.reduction])
            l_inst.run()
            tune_df.at[n] = l_inst.eigen_score
            cv_df.at[n] = [l_inst.cv_score, l_inst.boost_time]
            re_df.at[n] = l_inst.reconstruction
            inst_list[n] = l_inst

        # Create plots
        fig, ax = plt.subplots()
        ax.set_title("Total Variance over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("Variance")
        tune_df.reset_index().plot(kind='line', ax=ax,
                                   x='index', y='Variance Ratio', marker="o")

        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_pca_tuning.png"))

        # Create score plot
        fig, ax = plt.subplots()
        ax.set_title("Cross Val over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("CV")
        cv_df.reset_index().plot(kind='line', ax=ax, label="CV",
                                 x='index', y='CV', marker="o")
        ax2 = ax.twinx()
        ax2.set_ylabel("Time")
        cv_df.reset_index().plot(kind='line', ax=ax2, label="Time",
                                 x='index', y='Time', color='red', marker="o")
        ax.grid(False)
        ax.legend(loc="lower right")
        ax2.legend(loc="lower center")

        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_cross_validation.png"))

        # Create plots
        fig, ax = plt.subplots()
        ax.set_title("Reconstruction Error over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("Reconstruction Error")
        re_df.reset_index().plot(kind='line', ax=ax,
                                 x='index', y='Reconstruction Error')

        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_rp_tuning.png"))

    def ica_tuning(self, nmin, nmax):
        """ Tune n_components for ICA """
        # Init range and instances
        nrange = range(nmin, nmax)
        inst_list = dict()
        tune_df = pd.DataFrame(index=nrange, columns=['Total Kurtosis'])
        cv_df = pd.DataFrame(index=nrange, columns=['CV', 'Time'])
        re_df = pd.DataFrame(index=nrange, columns=['Reconstruction Error'])

        # Generate data
        for n in nrange:
            l_inst = self.generate_reducer(self.reduction, self.data_class, n, self.config[self.reduction])
            l_inst.run()
            tune_df.at[n] = l_inst.kurtosis
            cv_df.at[n] = [l_inst.cv_score, l_inst.boost_time]
            re_df.at[n] = l_inst.reconstruction
            inst_list[n] = l_inst

        # Create plots
        fig, ax = plt.subplots()
        ax.set_title("Total Kurtosis over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("Kurtosis")
        tune_df.reset_index().plot(kind='line', ax=ax,
                                   x='index', y='Total Kurtosis')

        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_ica_tuning.png"))

        # Create score plot
        fig, ax = plt.subplots()
        ax.set_title("Cross Val over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("CV")
        cv_df.reset_index().plot(kind='line', ax=ax, label="CV",
                                 x='index', y='CV', marker="o")
        ax2 = ax.twinx()
        ax2.set_ylabel("Time")
        cv_df.reset_index().plot(kind='line', ax=ax2, label="Time",
                                 x='index', y='Time', color='red', marker="o")
        ax.grid(False)
        ax.legend(loc="lower right")
        ax2.legend(loc="lower center")

        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_cross_validation.png"))

        # Create plots
        fig, ax = plt.subplots()
        ax.set_title("Reconstruction Error over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("Reconstruction Error")
        re_df.reset_index().plot(kind='line', ax=ax,
                                   x='index', y='Reconstruction Error')

        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_rp_tuning.png"))

    def rp_tuning(self, nmin, nmax):
        """ Tune n_components for ICA """
        # Init range and instances
        nrange = range(nmin, nmax)
        inst_list = dict()
        tune_df = pd.DataFrame(index=nrange, columns=['Reconstruction Error'])
        cv_df = pd.DataFrame(index=nrange, columns=['CV', 'Time'])

        # Generate data
        for n in nrange:
            l_inst = self.generate_reducer(self.reduction, self.data_class, n, self.config[self.reduction])
            l_inst.run()
            tune_df.at[n] = l_inst.reconstruction
            cv_df.at[n] = [l_inst.cv_score, l_inst.boost_time]
            inst_list[n] = l_inst

        # Create plots
        fig, ax = plt.subplots()
        ax.set_title("Reconstruction Error over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("Reconstruction Error")
        tune_df.reset_index().plot(kind='line', ax=ax,
                                   x='index', y='Reconstruction Error')

        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_rp_tuning.png"))

        # Create score plot
        fig, ax = plt.subplots()
        ax.set_title("Cross Val over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("CV")
        cv_df.reset_index().plot(kind='line', ax=ax, label="CV",
                                 x='index', y='CV', marker="o")
        ax2 = ax.twinx()
        ax2.set_ylabel("Time")
        cv_df.reset_index().plot(kind='line', ax=ax2, label="Time",
                                 x='index', y='Time', color='red', marker="o")
        ax.grid(False)
        ax.legend(loc="lower right")
        ax2.legend(loc="lower center")

        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_cross_validation.png"))

    def kpca_tuning(self, nmin, nmax):
        """ Tune n_components for Kernel PCA """
        # Init range and instances
        nrange = range(nmin, nmax)
        inst_list = dict()
        tune_df = pd.DataFrame(index=nrange, columns=['Variance'])
        cv_df = pd.DataFrame(index=nrange, columns=['CV', 'Time'])
        #re_df = pd.DataFrame(index=nrange, columns=['Reconstruction Error'])

        # Generate data
        for n in nrange:
            l_inst = self.generate_reducer(self.reduction, self.data_class, n, self.config[self.reduction])
            l_inst.run()
            tune_df.at[n] = l_inst.eigen_score
            cv_df.at[n] = [l_inst.cv_score, l_inst.boost_time]
            #re_df.at[n] = l_inst.reconstruction
            inst_list[n] = l_inst

        # Create plots
        fig, ax = plt.subplots()
        ax.set_title("Variance over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("Variance")
        tune_df.reset_index().plot(kind='line', ax=ax,
                                   x='index', y='Variance', marker="o")

        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_kpca_tuning.png"))

        # Create score plot
        fig, ax = plt.subplots()
        ax.set_title("Cross Val over n_components")
        ax.set_xlabel("n_components")
        ax.set_ylabel("CV")
        cv_df.reset_index().plot(kind='line', ax=ax, label="CV",
                                 x='index', y='CV', marker="o")
        ax2 = ax.twinx()
        ax2.set_ylabel("Time")
        cv_df.reset_index().plot(kind='line', ax=ax2, label="Time",
                                 x='index', y='Time', color='red', marker="o")
        ax.grid(False)
        ax.legend(loc="lower right")
        ax2.legend(loc="lower center")

        # Format and output all graphs
        fig.savefig(os.path.join(self.out_dir, f"{self.name}_cross_validation.png"))

        # Create plots
        #fig, ax = plt.subplots()
        #ax.set_title("Reconstruction Error over n_components")
        #ax.set_xlabel("n_components")
        #ax.set_ylabel("Reconstruction Error")
        #re_df.reset_index().plot(kind='line', ax=ax,
        #                           x='index', y='Reconstruction Error')

        # Format and output all graphs
        #fig.savefig(os.path.join(self.out_dir, f"{self.name}_rp_tuning.png"))

    def run(self):
        """ Run the experiment """
        print(f"Running {self.type} for part {self.part}")
        results = list()

        # Create Boosting Learner for scores
        trees = tree.DecisionTreeClassifier(max_depth=self.config["Boosting"]["max_depth"])
        self.boost = ensemble.AdaBoostClassifier(base_estimator=trees,
                                                 learning_rate=self.config["Boosting"]["learning_rate"],
                                                 n_estimators=self.config["Boosting"]["n_estimators"],
                                                 random_state=RANDOM_SEED)

        # Check the part of the project to run
        if self.part == "0":
            # Part 0 just means plot data analysis
            self.plot_data()

        # Part one can be tuning or plotting
        elif self.part == "1":
            # Generate instances
            self.process_instances()

            # Check the type of experiment
            if self.type == "plots":
                if self.cluster_class:
                    self.cluster_class.run()
                    self.cluster_class.plot()
                    results = results.append(self.cluster_class)
                    print(f'CV Score is {self.cluster_class.cv_score}')
                    print(f'Fit Time is {self.cluster_class.fit_time}')
                elif self.reduction_class:
                    self.reduction_class.run()
                    self.reduction_class.plot()
                    results = results.append(self.reduction_class)
                    print(f'CV Score is {self.reduction_class.cv_score}')
                    print(f'Fit Time is {self.reduction_class.fit_time}')

            elif self.type == "tuning":
                # Generate tuning plots
                if self.cluster:
                    if self.cluster == "kmeans":
                        self.plot_kelbow(self.config[self.cluster]["min"], self.config[self.cluster]["max"])
                        self.km_tuning(self.config[self.cluster]["min"], self.config[self.cluster]["max"])
                    elif self.cluster == "em":
                        self.em_tuning(self.config[self.cluster]["min"], self.config[self.cluster]["max"])
                else:
                    if self.reduction == "pca":
                        self.pca_tuning(self.config[self.reduction]["min"], self.config[self.reduction]["max"])
                    elif self.reduction == "ica":
                        self.ica_tuning(self.config[self.reduction]["min"], self.config[self.reduction]["max"])
                    elif self.reduction == "rp":
                        self.rp_tuning(self.config[self.reduction]["min"], self.config[self.reduction]["max"])
                    elif self.reduction == "kpca":
                        self.kpca_tuning(self.config[self.reduction]["min"], self.config[self.reduction]["max"])

        # Part two is combinations of DR then Cluster
        elif self.part == "2":
            # Generate instances
            self.process_instances()

            # Run DR and transform data
            self.reduction_class.run()
            l_data = self.reduction_class.dataset
            l_data.x_train = self.reduction_class.x_transform
            test_transform = self.reduction_class.instance.transform(l_data.x_test)
            l_data.x_test = test_transform

            # Save new data set class
            self.data_class = l_data

            # Run plots or tuning
            if self.type == "plots":
                # Run cluster class
                self.cluster_class.run()
                self.cluster_class.plot()
                print(f'CV Score is {self.cluster_class.cv_score}')
                print(f'Fit Time is {self.cluster_class.fit_time}')
            elif self.type == "tuning":
                if self.cluster:
                    if self.cluster == "kmeans":
                        self.plot_kelbow(self.config[self.cluster]["min"], self.config[self.cluster]["max"])
                        self.km_tuning(self.config[self.cluster]["min"], self.config[self.cluster]["max"])
                    elif self.cluster == "em":
                        self.em_tuning(self.config[self.cluster]["min"], self.config[self.cluster]["max"])

        # Part 3 is part 1 with Neural Networks
        elif self.part == "3":
            # Generate instances
            self.process_instances()

            # Run cluster if clustering
            if self.cluster:
                # Run cluster and setup data
                self.cluster_class.run()
                l_data = self.cluster_class.dataset
                l_data.x_train = l_data.x_cluster
                # Get cluster data for test
                test_clusters = self.cluster_class.instance.predict(l_data.x_test)
                df = pd.DataFrame(l_data.x_test)
                df["Cluster"] = test_clusters
                l_data.x_test = df

                # Run Neural Net
                self.neural_class = nn.NNClass(self.name, self.cluster_class.dataset,
                                               self.config["NN"], self.out_dir)
                self.neural_class.run()
                self.neural_class.plot_lca()
                self.neural_class.plot_loss()

            # Otherwise run reduction
            else:
                # Run DR and get data
                self.reduction_class.run()
                l_data = self.reduction_class.dataset
                l_data.x_train = self.reduction_class.x_transform
                test_transform = self.reduction_class.instance.transform(l_data.x_test)
                l_data.x_test = test_transform

                # Run Neural Net
                self.neural_class = nn.NNClass(self.name, l_data,
                                               self.config["NN"], self.out_dir)
                self.neural_class.run()
                self.neural_class.plot_lca()
                self.neural_class.plot_loss()

            # Print Metrics
            print(f'NN Complete')
            print(f'Train Acc: {self.neural_class.acc_train}')
            print(f'Test Acc: {self.neural_class.acc_test}')
            print(f'Loss: {self.neural_class.loss}')

        return results
