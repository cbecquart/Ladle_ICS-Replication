import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


n_comp_colors_1 = {
    "0": "#FFC9C9",
    "1": "#AED4B5",
    "2": "#96F7E4",
    "3": "#B8E6FE",
    "4": "#FAE8FF",
    "5": "#F6CFFF",
    ">=6": "#E4E4E7",
}

n_comp_colors_2 = {
    "0": "#FFC9C9",
    "1": "#FFD6A7",
    "2": "#AED4B5",
    "3": "#B8E6FE",
    "4": "#FAE8FF",
    "5": "#F6CFFF",
    ">=6": "#E4E4E7",
}

n_comp_colors_3 = {
    "0": "#FFC9C9",
    "1": "#FFD6A7",
    "2": "#96F7E4",
    "3": "#AED4B5",
    "4": "#FAE8FF",
    "5": "#F6CFFF",
    ">=6": "#E4E4E7",
}


def plot_stacked_counts(df, scatter_order=None, target_label="n_comp", pdf_name="ladle_counts_k"):
    """
    df must contain:
        k, model_index, scatters, n_boot. Counts are computed within the function.
    """

    sns.set_context("paper", rc={
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12
    })

    for k_val in sorted(df["k"].unique()):

        df_q = df[df["k"] == k_val].copy()

        if k_val == 2:
            n_comp_colors = n_comp_colors_1
        elif k_val == 3:
            n_comp_colors = n_comp_colors_2
        else:
            n_comp_colors = n_comp_colors_3

        # optional ordering for scatters
        if scatter_order is not None:
            df_q["scatters"] = pd.Categorical(
                df_q["scatters"],
                categories=scatter_order,
                ordered=True
            )

        # create the combined label for the Y-axis rows (Model then Scatter)
        df_q = df_q.sort_values(by=["model_index", "scatters"])
        df_q["model_scatters"] = df_q["model_index"].astype(str) + " x " + df_q["scatters"].astype(str)

        num_rows = df_q["model_scatters"].nunique()
        grid_height = max(3.5, num_rows * 0.3)

        g = sns.FacetGrid(
            df_q,
            col="n_boot",
            sharex=True,
            sharey=True,
            height=grid_height,
            aspect=1.2
        )

        def draw_stacked(data, **kwargs):
            data = data.copy()
            data[target_label] = data[target_label].astype(int)
            data[target_label] = data[target_label].clip(upper=6)
            data[target_label] = data[target_label].astype(str)
            data[target_label] = data[target_label].replace("6", ">=6")

            # reshape for stacked bars
            df_counts = (
                data
                .groupby(["model_scatters", "k", "n_boot", target_label], sort=False)
                .size()
                .reset_index(name="count")
            )

            pivot = df_counts.pivot(
                index="model_scatters",
                columns=target_label,
                values="count"
            ).fillna(0)

            n_comp_order = list(n_comp_colors.keys())
            pivot = pivot.reindex(columns=n_comp_order, fill_value=0)

            # keep the row ordering matching the sorted dataframe
            row_order = data["model_scatters"].unique()
            pivot = pivot.reindex(row_order, fill_value=0)

            # stacked horizontal bars
            ax = plt.gca()

            pivot.plot(
                kind="barh",
                stacked=True,
                ax=ax,
                color=[n_comp_colors[n] for n in pivot.columns],
                legend=False
            )

            ax.invert_yaxis()

        g.map_dataframe(draw_stacked)

        g.set_titles("n_boot: {col_name}")
        g.set_axis_labels("Counts", "Model x Scatters")

        handles, labels = g.axes.flat[0].get_legend_handles_labels()

        g.figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            bbox_to_anchor=(0.5, -0.05)
        )

        g.figure.subplots_adjust(
            top=0.90,
            bottom=0.15,
            wspace=0.2,
            hspace=0.2
        )

        pdf_full_name = f"plots/{pdf_name}{k_val}.pdf"
        g.figure.savefig(pdf_full_name, bbox_inches="tight", dpi=300)
        print(f"Figure saved as : {pdf_full_name}")


files = glob.glob("results_boot/result_*.csv")
list_df = [pd.read_csv(f) for f in files]
df_res = pd.concat(list_df, ignore_index=True)
df_res = df_res[df_res["scatters"] != "cov_cov4_ref"]

plot_stacked_counts(df_res, scatter_order=None, target_label="n_comp", pdf_name="boot_counts_k")
