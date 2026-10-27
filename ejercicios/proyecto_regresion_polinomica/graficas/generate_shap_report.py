"""
generate_shap_report.py

Publication-quality SHAP report generator for scikit-learn pipelines.
"""

import os
from typing import Optional
from xml.parsers.expat import model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    BaggingRegressor   
)
from senfarming_ai_utils import display_func, display_func_val
    

class GenerateShapReport:
    """Generate publication-quality SHAP explainability reports.

    Expected pipeline:
        scaler -> selector -> model

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
    X : pandas.DataFrame
    X_all : pandas.DataFrame
    output_path : str
    title : str
    subtitle : str
    dpi : int
    max_display : int
    """

    def __init__(
        self,
        pipeline,
        columns_filter,
        X: pd.DataFrame,
        output_path: str = ".",
        title: str = "SHAP Explainability Report",
        subtitle: str = "",
        name: str = "",
        dpi: int = 300,
        max_display: int = 20,
    ):
        self.pipeline = pipeline
        self.columns_filter=columns_filter
        self.X_original = X.copy()
        self.output_path = output_path
        self.title = title
        self.subtitle = subtitle
        self.name = name
        self.dpi = dpi
        self.max_display = max_display
        

        os.makedirs(output_path, exist_ok=True)

        self._explainer = None
        self._values = None
        self._X = None
        self._X_sel = None
        self._model = None

        self._set_style()

    def _set_style(self):
        plt.style.use("default")
        plt.rcParams.update({
            "figure.figsize": (12, 8),
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

    def _decorate(self, subtitle: str = ""):
        plt.title(self.title)
        if self.subtitle:
            plt.suptitle(self.subtitle, fontsize=14, color="gray")
        else:
            plt.suptitle(subtitle, fontsize=14, color="gray")

    def _save(self, name: str):
        plt.subplots_adjust(top=0.90)
        base = os.path.join(self.output_path, name)
        plt.savefig(base + ".png", dpi=self.dpi,
                    bbox_inches="tight", facecolor="white")
        plt.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
        plt.savefig(base + ".svg", bbox_inches="tight", facecolor="white")
        plt.close()

    def prepare_data(self):
        display_func(0,"prepare_data")
        
        X = self.X_original.copy()

        if "scaler" in self.pipeline.named_steps:
            display_func(0,"prepare_data step 1")
            X = self.pipeline.named_steps["scaler"].transform(X)
        display_func(0,"prepare_data step 10")    
        if "preprocessor" in self.pipeline.named_steps:
            display_func(0,"prepare_data step 11")  
            X = self.pipeline.named_steps["preprocessor"].transform(X)
        display_func(0,"prepare_data step 20")  
        if "selector" in self.pipeline.named_steps:
            display_func(0,"prepare_data step 21")  
            selector = self.pipeline.named_steps["selector"]
            X = selector.transform(X)
            names = self.X_original.columns[selector.get_support()]
        else:
            display_func(0,"prepare_data step 31")  
            names = self.X_original.columns

        X = pd.DataFrame(X, columns=names)
        if self.columns_filter is not None:
            X_sel = X.drop(
                columns=self.columns_filter,
                errors="ignore"
            )
        else:
            X_sel = X.copy()

        model = self.pipeline.named_steps["model"]
        return model, X , X_sel

    def create_explainer(self, model, X,X_sel,status):
        display_func(0,"create_explainer")
        """
        Create the most appropriate SHAP explainer according to the estimator.
        """
        status = 0
        # Fast tree explainers
        if isinstance(
            model,
            (
                RandomForestRegressor,
                ExtraTreesRegressor,
                GradientBoostingRegressor,
                HistGradientBoostingRegressor,
            ),
        ):
            return shap.TreeExplainer(
                model,
                feature_perturbation="tree_path_dependent"
            )

        # AdaBoost
        if isinstance(model, AdaBoostRegressor):
            display_func(0,f"Using model: {self._model.__class__.__name__}:, TreeExplainer, Create SHAP explainer...")
            try:
                return shap.TreeExplainer(
                    model,
                    feature_perturbation="tree_path_dependent"
                )

            except Exception as e:

                status = 1
                display_func(0,f"Using model: {self._model.__class__.__name__}: exception, Create SHAP explainer... status: {status}")
                display_func_val(0,f"Using model: {self._model.__class__.__name__}: exception, Create SHAP explainer...: ", str(e))
                return shap.Explainer(
                    model.predict,
                    X_sel
                )

        # Bagging
        if isinstance(model, BaggingRegressor):
            display_func(0,f"Using model: {self._model.__class__.__name__}: ,PermutationExplainer, Create SHAP explainer...")
            return shap.PermutationExplainer(
                model.predict,
                X_sel
            )
            
        # Generic estimator
        return shap.Explainer(
            model.predict,
            X
        )
    def compute(self):
        display_func(0,"compute")
        if self._values is not None:
                return self._explainer, self._values, self._X

        self._model, self._X, self._X_sel= self.prepare_data()
        status = 0
        print(f"Using model: {self._model.__class__.__name__}: compute Create SHAP explainer...")
        self._explainer = self.create_explainer(
            self._model,
            self._X,
            self._X_sel,
            status
        )
        

        try:
            display_func(0,f"Using model: {self._model.__class__.__name__}: Computing SHAP values...")
            if isinstance(self._model, AdaBoostRegressor):
                self._values = self._explainer(
                    self._X_sel.values,
                    check_additivity=False
                )
            if isinstance(self._model, BaggingRegressor):
                self._values = self._explainer(
                    self._X_sel.values,
                    check_additivity=False
                )
            else:
                self._values = self._explainer(
                    self._X ,
                    check_additivity=False
                )

        except TypeError:
            display_func(0,f"Using model: {self._model.__class__.__name__}: Computing SHAP values...TypeError")
            # TreeExplainer for older SHAP versions
            self._values = self._explainer(self._X_sel)

        except Exception:
            display_func(0,f"Using model: {self._model.__class__.__name__}: Computing SHAP values...Exception")
            # Kernel/Permutation explainers
            self._values = self._explainer(self._X)
        display_func(0,f"Using model: {self._model.__class__.__name__}: Computing SHAP values... shape values:{self._values.shape}")
        return self._explainer, self._values, self._X,self._X_sel

    def export_importance(self):
        display_func(0,"export_importance")
        if isinstance(self._model, AdaBoostRegressor):
            X_final = self._X_sel.copy()
        elif isinstance(self._model, BaggingRegressor):
            X_final = self._X_sel.copy()
        else:
            X_final = self._X.copy()
        imp = np.abs(self._values.values).mean(axis=0)
        df = pd.DataFrame(
            {"feature": X_final.columns, "importance": imp}
        ).sort_values("importance", ascending=False)
        df.to_csv(os.path.join(self.output_path,
                               "shap_importance.csv"), index=False)
        return df

    def summary_plot(self):
        display_func(0,"summary_plot")
        _, values, X , X_sel = self.compute()
        if isinstance(self._model, AdaBoostRegressor):
            X_final = X_sel.copy()
        elif isinstance(self._model, BaggingRegressor):
            X_final = X_sel.copy()
        else:
            X_final = X.copy()
        display_func(0,f"summary_plot, values shape: {values.shape}")
        display_func(0,f"summary_plot, X_final shape: {X_final.shape}")
        shap.summary_plot(
            values,
            X_final,
            max_display=self.max_display,
            show=False,
            plot_size=(12, 8),
        )
        plt.xlabel("SHAP value")
        self._decorate("Summary plot of SHAP values for selected features")
        self._save("shap_summary")

    def beeswarm(self):
        display_func(0,"beeswarm")
        shap.plots.beeswarm(self._values,
                            max_display=self.max_display,
                            show=False)
        self._decorate("Beeswarm plot of SHAP values for selected features")
        self._save("shap_beeswarm")

    def bar_plot(self):
        display_func(0,"bar_plot")
        df = self.export_importance().head(self.max_display)
        plt.figure(figsize=(10, 8))
        plt.barh(df.feature[::-1], df.importance[::-1])
        plt.xlabel("Mean |SHAP value|")
        plt.ylabel("Feature")
        self._decorate("Bar plot of mean absolute SHAP values for selected features")
        self._save("shap_bar")

    def waterfall(self, sample: int = 0):
        display_func(0,"waterfall")
        shap.plots.waterfall(self._values[sample],
                             max_display=self.max_display,
                             show=False)
        self._decorate("Waterfall plot of SHAP values for selected features")
        self._save(f"shap_waterfall_{sample}")

    def dependence(self, feature):
        display_func(0,"dependence")
        shap.plots.scatter(self._values[:, feature], alpha=0.7, show=False)
        self._decorate(f"Dependence plot of SHAP values for feature '{feature}'")
        self._save(f"shap_dependence_{feature}")

    def decision_plot(self, sample: int = 0):
        display_func(0,"decision_plot")
        if isinstance(self._model, AdaBoostRegressor):
            X_final = self._X_sel.copy()
        elif isinstance(self._model, BaggingRegressor):
            X_final = self._X_sel.copy()
        else:
            X_final = self._X.copy()
        display_func(0,f"decision_plot, values shape: {self._values.shape}")
        display_func(0,f"decision_plot, X_final shape: {X_final.shape}")
        try:
            plt.figure(figsize=(12, 8))
            shap.decision_plot(
                self._explainer.expected_value,
                self._values.values[sample],
                X_final.columns,
                show=False
            )
            self._decorate(f"Decision plot of SHAP values for sample {sample}")
            self._save(f"shap_decision_{sample}")
        except Exception as e:
            print("Decision plot skipped:", e)

    def heatmap(self):
        display_func(0,"heatmap")
        try:
            shap.plots.heatmap(self._values,
                               max_display=self.max_display,
                               show=False)
            self._decorate("Heatmap of SHAP values for selected features")
            self._save("shap_heatmap")
        except Exception as e:
            print("Heatmap skipped:", e)

    def force_plot(self, sample: int = 0):
        display_func(0,"force_plot")
        if isinstance(self._model, AdaBoostRegressor):
            X_final = self._X_sel.copy()
        elif isinstance(self._model, BaggingRegressor):
            X_final = self._X_sel.copy()
        else:
            X_final = self._X.copy()
        display_func(0,f"force_plot, values shape: {self._values.shape}")
        display_func(0,f"force_plot, X_final shape: {X_final.shape}")
        try:
            force = shap.force_plot(
                self._explainer.expected_value,
                self._values.values[sample],
                X_final.iloc[sample],
                matplotlib=False
            )
            shap.save_html(
                os.path.join(self.output_path,
                             f"shap_force_{sample}.html"),
                force,
            )
        except Exception as e:
            print("Force plot skipped:", e)

    def generate(self,
                 dependence_top: int = 5,
                 waterfall_sample: int = 0):

        print("=" * 60)
        print("Generating SHAP Report")
        print("=" * 60)
        print("Summary plot...")
        self.summary_plot()
        print("Beeswarm plot...")
        self.beeswarm()
        print("Bar plot...")
        self.bar_plot()
        print("Waterfall plot...")
        self.waterfall(waterfall_sample)
        #print("Decision plot...")       
        #self.decision_plot(waterfall_sample)
        print("Heatmap...")
        self.heatmap()
        print("Force plot...")
        self.force_plot(waterfall_sample)

        #df = self.export_importance()
        #print("Dependence plots...")
        #for feature in df.feature.head(dependence_top):
        #    try:
        #        self.dependence(feature)
        #    except Exception as e:
        #        print(f"Dependence plot for {feature} skipped:", e)

        print("Done.")
        print(f"Results saved in: {self.output_path}")


if __name__ == "__main__":
    print("Import this class into your ML project.")
