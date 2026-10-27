import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RandomizedSearchVisualizer:

    def __init__(self,
                 search,
                 output_path="random_search_report",
                 dpi=300):

        self.search = search
        self.output_path = output_path
        self.dpi = dpi

        os.makedirs(output_path, exist_ok=True)

        self.df = pd.DataFrame(search.cv_results_)

    ############################################################

    def generate(self):

        self.save_results()

        self.plot_best_score_progress()

        self.plot_score_distribution()

        self.plot_score_boxplot()

        params = self.get_parameter_names()

        if len(params) >= 2:
            self.plot_parameter_scatter(params[0], params[1])

        if len(params) >= 2:
            self.plot_heatmap(params[0], params[1])

    ############################################################

    def save_results(self):

        self.df.to_csv(
            os.path.join(self.output_path,
                         "randomized_search_results.csv"),
            index=False)

    ############################################################

    def get_parameter_names(self):

        names = []

        for c in self.df.columns:

            if c.startswith("param_"):
                names.append(c)

        return names

    ############################################################

    def plot_best_score_progress(self):

        scores = self.df["mean_test_score"].values

        best = []

        current = -np.inf

        for s in scores:

            current = max(current, s)

            best.append(current)

        plt.figure(figsize=(8,5))

        plt.plot(
            np.arange(1, len(best)+1),
            best,
            linewidth=2)

        plt.xlabel("Iteration")
        plt.ylabel("Best CV Score")
        plt.title("Randomized Search Convergence")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_path,
                "01_convergence.png"),
            dpi=self.dpi)

        plt.close()

    ############################################################

    def plot_score_distribution(self):

        plt.figure(figsize=(7,5))

        plt.hist(
            self.df["mean_test_score"],
            bins=20)

        plt.xlabel("Mean CV Score")

        plt.ylabel("Frequency")

        plt.title("Distribution of Scores")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_path,
                "02_histogram.png"),
            dpi=self.dpi)

        plt.close()

    ############################################################

    def plot_score_boxplot(self):

        plt.figure(figsize=(4,6))

        plt.boxplot(
            self.df["mean_test_score"],
            vert=True)

        plt.ylabel("Mean CV Score")

        plt.title("Score Distribution")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_path,
                "03_boxplot.png"),
            dpi=self.dpi)

        plt.close()

    ############################################################

    def plot_parameter_scatter(self,
                               param1,
                               param2):

        df = self.df.copy()

        try:

            x = pd.to_numeric(df[param1])

            y = pd.to_numeric(df[param2])

        except:

            return

        plt.figure(figsize=(8,6))

        scatter = plt.scatter(
            x,
            y,
            c=df["mean_test_score"],
            s=80)

        plt.xlabel(param1.replace("param_",""))

        plt.ylabel(param2.replace("param_",""))

        plt.title("Random Search Samples")

        plt.colorbar(scatter,
                     label="Mean CV Score")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_path,
                "04_parameter_scatter.png"),
            dpi=self.dpi)

        plt.close()

    ############################################################

    def plot_heatmap(self,
                     param1,
                     param2):

        df = self.df.copy()

        try:

            df[param1] = pd.to_numeric(df[param1])

            df[param2] = pd.to_numeric(df[param2])

        except:

            return

        pivot = df.pivot_table(
            values="mean_test_score",
            index=param2,
            columns=param1,
            aggfunc=np.mean)

        plt.figure(figsize=(8,6))

        plt.imshow(
            pivot,
            aspect="auto",
            origin="lower")

        plt.colorbar(label="Mean CV Score")

        plt.xticks(
            np.arange(len(pivot.columns)),
            pivot.columns,
            rotation=45)

        plt.yticks(
            np.arange(len(pivot.index)),
            pivot.index)

        plt.xlabel(param1.replace("param_",""))

        plt.ylabel(param2.replace("param_",""))

        plt.title("Performance Heatmap")

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_path,
                "05_heatmap.png"),
            dpi=self.dpi)

        plt.close()