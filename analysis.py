"""
Module: analysis.py

### About This Module
This module analyzes prompt optimization experiment results and generates visualizations.

### Key Features
- Loads summary and mutation results from experiment CSV files.
- Plots optimization score evolution across generations.
- Compares baseline and optimized test scores across models.
- Computes mutation improvement rates for each optimization branch.

### Functions
- load_results: Loads and combines experiment result CSV files.
- plot_optimization_evolution: Plots optimization score evolution.
- plot_test_scores: Compares baseline and optimized test scores.
- parse_parent_scores: Parses parent mutation scores.
- prepare_mutation_metrics: Computes mutation improvement status.
- plot_improvement_rate: Plots improvement rates by optimization branch.
"""

import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import komorebi_mpl

'''plt.style.use("night_wave")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 14,
    }
)'''

RESULTS_DIR = "results"
ANALYSIS_DIR = "analysis"


def truncate_model_name(model):
    """
    Truncates a model name after its parameter specification.

    Args:
        model: Model name or identifier.

    Returns:
        str: Truncated model name.
    """

    match = re.search(
        r"^(.+?-\d+(?:\.\d+)?B)",
        str(model),
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return str(model)


def load_results(results_dir=RESULTS_DIR):
    """
    Loads summary and mutation CSV files from the experiments results directory.

    Args:
        results_dir (str, optional): Directory containing experiment results.

    Returns:
        pandas.DataFrame: Combined summary and mutation results.
    """

    records = []

    # Scan result directories.
    for root, _, files in os.walk(results_dir):

        for filename in files:

            # Identify result level from filename.
            if filename.endswith("_summary.csv"):
                level = "summary"

            elif filename.endswith("_mutations.csv"):
                level = "mutation"

            else:
                continue

            path = os.path.join(root, filename)

            # Load CSV file.
            try:
                df = pd.read_csv(path)

            except Exception as exc:
                print(f"Could not read {path}: {exc}")
                continue

            if df.empty:
                continue

            # Extract task and model from directory structure.
            parts = os.path.normpath(root).split(os.sep)

            if len(parts) < 2:
                continue

            model = parts[-1]
            task = parts[-2]

            # Add metadata to each result row.
            for _, row in df.iterrows():

                record = row.to_dict()

                record["model"] = model
                record["task"] = task
                record["source_file"] = path
                record["level"] = level

                records.append(record)

    results = pd.DataFrame(records)

    if results.empty:
        return results

    # Convert numeric columns.
    numeric_columns = [
        "generation",
        "optimization_score",
        "test_score",
        "exact",
        "judge",
        "child_score",
        "score",
    ]

    for column in numeric_columns:

        if column in results.columns:

            results[column] = pd.to_numeric(
                results[column],
                errors="coerce",
            )

    return results

# Optimization-set performance.
def plot_optimization_evolution(
    results,
    output_dir=ANALYSIS_DIR,
):
    """
    Creates optimization score evolution plots.

    Args:
        results (pandas.DataFrame): Combined experiment results.
        output_dir (str, optional): Directory for generated plots.
    """

    output_dir = os.path.join(
        output_dir,
        "optimization_evolution",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # Select optimization summary results.
    optimization = results[
        (results["level"] == "summary")
        & (results["optimizer"] != "none")
        & results["optimization_score"].notna()
    ].copy()

    groups = optimization.groupby(["task", "prompt"])

    for (task, prompt), group in groups:

        models = sorted(group["model"].unique())

        n_models = len(models)

        fig, axes = plt.subplots(
            n_models,
            1,
            figsize=(11, 5 * n_models),
            squeeze=False,
        )

        axes = axes.flatten()

        legend_handles = []
        legend_labels = []

        for ax, model in zip(axes, models):

            model_data = group[group["model"] == model]

            optimizers = sorted(model_data["optimizer"].unique())

            # Plot each optimization method.
            for optimizer in optimizers:

                data = model_data[model_data["optimizer"] == optimizer].sort_values(
                    "generation"
                )

                if data.empty:
                    continue

                (line,) = ax.plot(
                    data["generation"],
                    data["optimization_score"],
                    marker="o",
                    label=optimizer,
                )

                if optimizer not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(optimizer)

            ax.set_title(truncate_model_name(model), fontweight="bold")

            ax.set_xlabel("Optimization generation", fontweight="bold")

            ax.set_ylabel("Optimization score", fontweight="bold")

            ax.grid(
                True,
                alpha=0.3,
            )

            ax.set_facecolor("whitesmoke")

        # Use the overall score range for every subplot.
        overall_min = group["optimization_score"].min()
        overall_max = group["optimization_score"].max()

        for ax in axes:
            ax.set_ylim(
                overall_min - 0.01,
                overall_max + 0.01,
            )

        fig.suptitle(
            f"{task} | Initial prompt: {prompt}",
            fontsize=16,
            fontweight="bold",
        )

        # Create one legend for the entire figure.
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=len(legend_labels),
            title="Optimization methods",
            title_fontsize=18,
            fontsize=16,
            fancybox=True,
            shadow=True,
        )

        fig.tight_layout(rect=[0, 0.06, 1, 0.97])

        filename = f"{task}_{prompt}_optimization_evolution.png"

        fig.savefig(
            os.path.join(
                output_dir,
                filename,
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

# Test-set performance.
def plot_test_scores(
    results,
    output_dir=ANALYSIS_DIR,
):
    """
    Creates test score comparison plots.
    The plot compares baseline and optimized scores across models.

    Args:
        results (pandas.DataFrame): Combined experiment results.
        output_dir (str, optional): Directory for generated plots.
    """

    output_dir = os.path.join(
        output_dir,
        "test_scores",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # Select summary results with test scores.
    summaries = results[
        (results["level"] == "summary") & results["test_score"].notna()
    ].copy()

    groups = summaries.groupby(["task", "prompt"])

    for (task, prompt), group in groups:

        models = sorted(group["model"].unique())

        fig, ax = plt.subplots(figsize=(9, 7))

        # Define horizontal model positions.
        x_spacing = 0.65

        model_positions = {model: i * x_spacing for i, model in enumerate(models)}

        legend_handles = []
        legend_labels = []

        # Select optimized results.
        optimizer_data = group[group["optimizer"] != "none"]

        optimizers = sorted(optimizer_data["optimizer"].unique())

        # Plot optimized test scores.
        for optimizer in optimizers:

            data = optimizer_data[optimizer_data["optimizer"] == optimizer]

            x_values = []
            y_values = []

            for model in models:

                model_data = data[data["model"] == model]

                if model_data.empty:
                    continue

                x_values.append(model_positions[model])

                y_values.append(model_data.iloc[0]["test_score"])

            if x_values:

                scatter = ax.scatter(
                    x_values,
                    y_values,
                    label=optimizer,
                    s=70,
                )

                legend_handles.append(scatter)
                legend_labels.append(optimizer)

        # Select baseline results.
        baseline_data = group[group["optimizer"] == "none"]

        baseline_x = []
        baseline_y = []

        for model in models:

            model_baseline = baseline_data[baseline_data["model"] == model]

            if model_baseline.empty:
                continue

            baseline_x.append(model_positions[model])

            baseline_y.append(model_baseline.iloc[0]["test_score"])

        # Plot baseline test scores.
        if baseline_x:

            baseline_scatter = ax.scatter(
                baseline_x,
                baseline_y,
                label="baseline",
                s=180,
                marker="D",
                zorder=5,
            )

            legend_handles.append(baseline_scatter)
            legend_labels.append("baseline")

        # Use the overall score range shown in the figure.
        overall_min = group["test_score"].min()
        overall_max = group["test_score"].max()

        ax.set_ylim(
            overall_min - 0.01,
            overall_max + 0.01,
        )

        ax.set_title(
            f"{task} | Initial prompt: {prompt}",
            fontweight="bold",
            fontsize=16,
        )

        ax.set_xlabel("Model", fontweight="bold")

        ax.set_ylabel("Test score", fontweight="bold")

        ax.set_xticks([model_positions[model] for model in models])

        ax.set_xticklabels(
            [truncate_model_name(model) for model in models],
            rotation=20,
            ha="right",
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        ax.set_facecolor("whitesmoke")

        # Create one legend for the entire figure.
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=len(legend_labels),
            title="Optimization methods",
            fancybox=True,
            shadow=True,
        )

        fig.tight_layout(rect=[0, 0.08, 1, 1])

        filename = f"{task}_{prompt}_test_scores.png"

        fig.savefig(
            os.path.join(
                output_dir,
                filename,
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def parse_parent_scores(value):
    """
    Parses parent scores into a list of numeric values.

    Ordinary mutation:
        x to [x]

    Crossover mutation:
        "(x, y)" to [x, y]

    Args:
        value: Parent score value from a mutation result.

    Returns:
        list: Parsed parent scores, or None when parsing fails.
    """

    if pd.isna(value):
        return None

    if isinstance(value, (int, float, np.number)):
        return [float(value)]

    if isinstance(value, (tuple, list)):
        try:
            scores = [float(x) for x in value]
            return scores if scores else None
        except (TypeError, ValueError):
            return None

    try:
        parsed = ast.literal_eval(str(value))

    except (ValueError, SyntaxError):
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return None

    if isinstance(parsed, (tuple, list)):
        try:
            scores = [float(x) for x in parsed]
            return scores if scores else None
        except (TypeError, ValueError):
            return None

    try:
        return [float(parsed)]
    except (TypeError, ValueError):
        return None


def prepare_mutation_metrics(mutations):
    """
    Computes whether each mutation improved its parent score.

    Ordinary mutation:
        child_score > parent_score

    Crossover:
        child_score > mean(parent_scores)

    Args:
        mutations (pandas.DataFrame): Mutation-level results.

    Returns:
        pandas.DataFrame: Mutation results with parsed parent scores
        and improvement status.
    """

    if mutations.empty:
        return mutations.copy()

    data = mutations.copy()

    # Parse parent scores.
    data["parent_scores"] = data["parent_score"].apply(parse_parent_scores)

    data["improved"] = np.nan

    for idx, row in data.iterrows():

        parent_scores = row["parent_scores"]
        child_score = row["child_score"]

        if parent_scores is None or pd.isna(child_score) or not parent_scores:
            continue

        branch = row["branch"]

        # Compare crossover children against the mean parent score.
        if branch == "crossover":

            parent_reference = sum(parent_scores) / 2

            improved = child_score > parent_reference

        # Compare ordinary mutations against their parent score.
        else:

            parent_reference = parent_scores[0]

            improved = child_score > parent_reference

        data.at[idx, "improved"] = int(improved)

    # Convert improvement status to numeric values.
    data["improved"] = pd.to_numeric(
        data["improved"],
        errors="coerce",
    )

    return data

# Mutation-level analysis.
def plot_improvement_rate(
    mutations,
    output_dir=ANALYSIS_DIR,
):
    """
    Creates mutation improvement rate plots visualizing the amount of child prompts
    that beat their parent in optimization score.

    Args:
        mutations (pandas.DataFrame): Mutation results with improvement metrics.
        output_dir (str, optional): Directory for generated plots.
    """

    output_dir = os.path.join(
        output_dir,
        "improvement_rate",
    )

    os.makedirs(output_dir, exist_ok=True)

    # Select mutations with valid improvement status.
    data = mutations[mutations["improved"].notna()].copy()

    groups = data.groupby(["task", "init_prompt"])

    for (task, prompt), group in groups:

        models = sorted(group["model"].unique())

        n_models = len(models)

        fig, axes = plt.subplots(
            n_models,
            1,
            figsize=(11, 4.5 * n_models),
            squeeze=False,
        )

        axes = axes.flatten()

        for ax, model in zip(axes, models):

            model_data = group[group["model"] == model]

            # Compute improvement statistics by branch.
            stats = (
                model_data.groupby("branch")["improved"]
                .agg(["mean", "sum", "count"])
                .sort_index()
            )

            rates = stats["mean"] * 100

            bars = ax.bar(rates.index, rates.values)

            # Add successful and total mutation counts.
            for bar, (_, row) in zip(bars, stats.iterrows()):

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{int(row['sum'])}/{int(row['count'])}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax.set_ylim(0, 110)

            ax.set_title(
                truncate_model_name(model),
                fontweight="bold",
            )

            ax.set_xlabel(
                "Optimization branch",
                fontweight="bold",
            )

            ax.set_ylabel(
                "Improvement rate (%)",
                fontweight="bold",
            )

            ax.grid(axis="y", alpha=0.3)

            ax.set_facecolor("whitesmoke")

        fig.suptitle(
            f"{task} | Initial prompt: {prompt}",
            fontsize=16,
            fontweight="bold",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        filename = f"{task}_{prompt}_improvement_rate.png"

        fig.savefig(
            os.path.join(output_dir, filename),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


# Load experiment results.
results = load_results(RESULTS_DIR)

# Generate optimization evolution plots.
plot_optimization_evolution(
    results,
    ANALYSIS_DIR,
)

# Generate test score comparison plots.
plot_test_scores(
    results,
    ANALYSIS_DIR,
)

# Compute mutation metrics.
mutations = prepare_mutation_metrics(results)

# Generate mutation improvement rate plots.
plot_improvement_rate(
    mutations,
    ANALYSIS_DIR,
)
