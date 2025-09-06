import os
import ast


def read_all_python_dicts_from_dir(directory):
    all_dicts = []

    for filename in os.listdir(directory):
        if 'pipeline' not in filename.lower():
            continue
        filepath = os.path.join(directory, filename)
        print(f"Reading {filename}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    # ast.literal_eval safely evaluates Python literal structures (dict, list, etc.)
                    obj = ast.literal_eval(line)
                    obj['file'] = filename
                    all_dicts.append(obj)
                except Exception as e:
                    print(f"Error parsing line {line_num} in {filename}: {e}")
    return all_dicts


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from typing import List, Dict, Any, Optional, Callable

metrics = [
    'em',
    'acc',
    'f1',
    'rouge1.precision',
    'rouge1.recall',
    'rouge1.fmeasure',
    'rouge2.precision',
    'rouge2.recall',
    'rouge2.fmeasure',
    'rougeL.precision',
    'rougeL.recall',
    'rougeL.fmeasure',
    'bertscore_precision',
    'bertscore_recall',
    'bertscore_f1',
]
variants = ['standard_predictions']


def where(data: List[Dict[str, Any]], **filters) -> List[Dict[str, Any]]:
    return [
        item for item in data
        if all(item.get(key) == value for key, value in filters.items())
    ]


def get_nested_value(data: Dict[str, Any], key_path: str) -> Any:
    keys = key_path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def plot_line(
        data: List[Dict[str, Any]],
        y_key_path: str | list,
        x_key_path: Optional[str] = None,
        title: Optional[str] = None
):
    if isinstance(y_key_path, str):
        y_key_path = [y_key_path]

    plots_per_row = 3
    n_metrics = len(y_key_path)
    n_rows = (n_metrics + plots_per_row - 1) // plots_per_row

    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(plots_per_row * 5, n_rows * 5))
    axes = axes.flatten()

    for i, metric_key in enumerate(y_key_path):
        extracted = []
        for j, item in enumerate(data):
            x_val = get_nested_value(item, x_key_path) if x_key_path else j
            y_val = get_nested_value(item, metric_key)
            if y_val is not None:
                extracted.append({"x": x_val, "y": y_val})

        df = pd.DataFrame(extracted)
        sns.lineplot(data=df, x="x", y="y", marker="o", linewidth=2, ax=axes[i])
        axes[i].set_title(metric_key)
        axes[i].set_xlabel(x_key_path if x_key_path else "Index")
        axes[i].set_ylabel(metric_key.split(".")[-1])
        axes[i].tick_params(axis='x', rotation=45)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if title:
        fig.suptitle(title, fontsize=14)
        plt.subplots_adjust(top=0.9)

    plt.tight_layout()
    plt.show()


def plot_bar(
        data: List[Dict[str, Any]],
        y_key_path: str,
        x_key_path: str,
        aggregate: Callable = pd.Series.mean
):
    extracted = []
    for item in data:
        x_val = get_nested_value(item, x_key_path)
        y_val = get_nested_value(item, y_key_path)
        if x_val is not None and y_val is not None:
            extracted.append({"x": x_val, "y": y_val})

    df = pd.DataFrame(extracted)

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=df,
        x="x",
        y="y",
        hue="x",  # set hue to x
        estimator=aggregate,
        errorbar=None,
        palette="pastel",
        legend=False  # disable legend
    )

    plt.title(f"{y_key_path} by {x_key_path}")
    plt.xlabel(x_key_path)
    plt.ylabel(y_key_path.split('.')[-1])
    plt.xticks()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()


def plot_all_metrics(data: List[Dict[str, Any]], x_key: Optional[str] = 'retrieval_topk', label: Optional[str] = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    plots_per_row = 3

    for variant in variants:
        if label:
            print(f"\n[{label.upper()} - {variant.upper()}]")
        else:
            print(f"\n[{variant.upper()}]")

        n_metrics = len(metrics)
        n_rows = (n_metrics + plots_per_row - 1) // plots_per_row

        fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(plots_per_row * 5, n_rows * 4))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            y_key = f"{variant}.{metric}"
            extracted = []
            for j, item in enumerate(data):
                x_val = get_nested_value(item, x_key) if x_key else j
                y_val = get_nested_value(item, y_key)
                if y_val is not None:
                    extracted.append({"x": x_val, "y": y_val})
            df = pd.DataFrame(extracted)

            sns.lineplot(data=df, x="x", y="y", marker="o", linewidth=2, ax=axes[i])
            axes[i].set_title(metric)
            axes[i].set_xlabel(x_key)
            axes[i].set_ylabel(metric.split('.')[-1])
            axes[i].tick_params(axis='x', rotation=45)

        # Hide any unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_grouped_metrics(
    data: List[Dict[str, Any]],
    group_key: str,
    x_key: str = 'retrieval_topk',
    label: Optional[str] = None,
    baseline_data: Optional[List[Dict[str, Any]]] = None
):
    plots_per_row = 3
    all_max = {}
    all_min = {}

    for variant in variants:
        if label:
            print(f"\n[{label.upper()} - {variant.upper()}]")
        else:
            print(f"\n[{variant.upper()}]")

        n_metrics = len(metrics)
        n_rows = (n_metrics + plots_per_row - 1) // plots_per_row

        fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(plots_per_row * 5, n_rows * 4))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            y_key = f"{variant}.{metric}"
            extracted = []

            mx = 0
            mn = 100
            groups = set()

            for item in data:
                group_val = get_nested_value(item, group_key)
                x_val = get_nested_value(item, x_key)
                y_val = get_nested_value(item, y_key)

                if group_val is not None and x_val is not None and y_val is not None:
                    mn = min(mn, y_val)
                    mx = max(mx, y_val)
                    extracted.append({
                        "x": x_val,
                        "y": y_val,
                        "group": group_val
                    })
                    groups.add(group_val)

            all_max[metric] = mx
            all_min[metric] = mn

            df = pd.DataFrame(extracted)

            # Get color palette and group-color mapping
            palette = sns.color_palette(n_colors=len(groups))
            group_to_color = {group: color for group, color in zip(sorted(groups), palette)}

            sns.lineplot(
                data=df,
                x="x",
                y="y",
                hue="group",
                palette=group_to_color,
                marker="o",
                linewidth=2,
                ax=axes[i]
            )

            # Add baseline dotted lines (same color as group)
            if baseline_data:
                for baseline in baseline_data:
                    baseline_group = get_nested_value(baseline, group_key)
                    baseline_y = get_nested_value(baseline, y_key)

                    if baseline_group is not None and baseline_y is not None:
                        color = group_to_color.get(baseline_group, "gray")
                        axes[i].axhline(
                            y=baseline_y,
                            linestyle='dotted',
                            linewidth=2,
                            color=color,
                            label=f"{baseline_group} (no RAG)"
                        )

            axes[i].set_title(metric)
            axes[i].set_xlabel(x_key)
            axes[i].set_ylabel(metric.split('.')[-1])
            axes[i].tick_params(axis='x', rotation=45)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        print(f"Max: {all_max}, Min: {all_min}")


def plot_all_heatmaps(
        data: List[Dict[str, Any]],
        x_key: str,
        y_key: str,
        label: Optional[str] = None
):
    plots_per_row = 3

    for variant in variants:
        if label:
            print(f"\n[{label.upper()} - {variant.upper()}]")
        else:
            print(f"\n[{variant.upper()}]")

        n_metrics = len(metrics)
        n_rows = (n_metrics + plots_per_row - 1) // plots_per_row

        fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(plots_per_row * 6, n_rows * 3))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            metric_key = f"{variant}.{metric}"

            extracted = []
            for item in data:
                x_val = get_nested_value(item, x_key)
                y_val = get_nested_value(item, y_key)
                z_val = get_nested_value(item, metric_key)
                if None not in (x_val, y_val, z_val):
                    extracted.append({
                        "x": x_val,
                        "y": y_val,
                        "z": z_val,
                    })

            df = pd.DataFrame(extracted)
            if df.empty:
                axes[i].set_visible(True)
                continue

            pivot = df.pivot(index="y", columns="x", values="z")
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[i])
            axes[i].set_title(metric)
            axes[i].set_xlabel(x_key)
            axes[i].set_ylabel(y_key)
            axes[i].tick_params(axis='x', rotation=45)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def plot_all_bar(
        data: List[Dict[str, Any]],
        x_key: str,
        label: Optional[str] = None,
        aggregate: Callable = pd.Series.mean
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import math

    plots_per_row = 3

    for variant in variants:
        if label:
            print(f"\n[{label.upper()} - {variant.upper()}]")
        else:
            print(f"\n[{variant.upper()}]")

        n_metrics = len(metrics)
        n_rows = math.ceil(n_metrics / plots_per_row)

        fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(plots_per_row * 5, n_rows * 5))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            y_key = f"{variant}.{metric}"
            extracted = []
            for item in data:
                x_val = get_nested_value(item, x_key)
                y_val = get_nested_value(item, y_key)
                if x_val is not None and y_val is not None:
                    extracted.append({"x": x_val, "y": y_val})

            df = pd.DataFrame(extracted)
            if df.empty:
                axes[i].set_visible(False)
                continue

            sns.barplot(
                data=df,
                x="x",
                y="y",
                hue="x",
                estimator=aggregate,
                errorbar=None,
                palette="pastel",
                legend=False,
                ax=axes[i]
            )
            axes[i].set_title(metric)
            axes[i].set_xlabel(x_key)
            axes[i].set_ylabel(metric.split('.')[-1])
            axes[i].tick_params(axis='x', rotation=0)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
