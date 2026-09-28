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
        k, model_index, scatters. Counts are computed within the function.
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
            model_order = ["GMM1", "SMM1_df3", "SMM8_df3", "PEM1"]
        elif k_val == 3:
            n_comp_colors = n_comp_colors_2
            model_order = ["GMM2", "GMM3", "GMM4", "GMM8", "GMM9", "SMM2_df3", "SMM3_df3", "SMM4_df3",
                           "PEM2", "PEM3", "PEM4", "PEM8", "PEM9", "PEM10", "PEM11"]
        else:
            n_comp_colors = n_comp_colors_3
            model_order = ["GMM5", "GMM6", "GMM7", "SMM5_df3", "SMM6_df3", "SMM7_df3", "PEM5", "PEM6", "PEM7"]

        # optional ordering for scatters
        if scatter_order is not None:
            df_q["scatters"] = pd.Categorical(
                df_q["scatters"],
                categories=scatter_order,
                ordered=True
            )

        g = sns.FacetGrid(
            df_q,
            col="scatters",
            col_wrap=3,
            sharex=True,
            sharey=True,
            height=3.5,
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
                .groupby(["model_index", "k", "scatters", target_label])
                .size()
                .reset_index(name="count")
            )

            pivot = df_counts.pivot(
                index="model_index",
                columns=target_label,
                values="count"
            ).fillna(0)

            n_comp_order = list(n_comp_colors.keys())
            pivot = pivot.reindex(columns=n_comp_order, fill_value=0)

            # impose the desired model order
            if model_order is not None:
                pivot = pivot.reindex(model_order, fill_value=0)

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

        g.figure.subplots_adjust(
            top=0.85,
            bottom=0.15
        )

        g.set_titles("{col_name}")
        g.set_axis_labels("Counts", "Model")

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
            bottom=0.12,
            wspace=0.2,
            hspace=0.2
        )

        pdf_full_name = f"plots/{pdf_name}{k_val}.pdf"
        g.figure.savefig(pdf_full_name, bbox_inches="tight", dpi=300)
        print(f"Figure saved as : {pdf_full_name}")

        plt.show()


files = glob.glob("results/result_*.csv")
list_df = [pd.read_csv(f) for f in files]
df_res = pd.concat(list_df, ignore_index=True)

plot_stacked_counts(df_res, scatter_order=None, target_label="n_comp", pdf_name="ladle_counts_k")

df_dftu = df_res[~(df_res["scatters"] == "cov_cov4_ref")]

plot_stacked_counts(df_dftu, scatter_order=None, target_label="dftu_ncomp", pdf_name="dftu_counts_k")

