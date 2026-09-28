import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_metrics(res):
    x = np.arange(res["p"])

    metrics = [
        ("fn", res["fn"], r"$\Huge f(r)$"),
        ("phin", res["phin"], r"$\Huge \phi(r)$"),
        ("gn", res["gn"], r"$\Huge g(r)$"),
    ]

    for name, values, latex_name in metrics:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode="lines+markers",
                name=name,
                line=dict(width=5, color="rgb(178, 79, 175)"),
                marker=dict(size=15, color="rgb(178, 79, 175)")
            )
        )

        fig.update_xaxes(
            title=dict(
                text=r"$\Huge r$",
                font=dict(size=42)
            ),
            range=[-0.2, res["p"] - 0.8],
            tickmode="array",
            tickvals=x,
            ticktext=[str(i) for i in x],
            tickfont=dict(size=25),
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True
        )

        fig.update_yaxes(
            title=dict(
                text=latex_name,
                font=dict(size=42)
            ),
            tickfont=dict(size=25),
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True
        )

        fig.update_layout(
            title="",
            width=600,
            height=500,
            template="plotly_white",
            showlegend=False,
            margin=dict(
                l=60,
                r=20,
                t=50,
                b=60
            )
        )

        fig.write_image(f"plots/ladleplot_{name}.pdf")


if __name__ == "__main__":
    results = []

    df_res = pd.read_csv("results/result_18.csv")
    model_res = df_res.iloc[[0]]

    fn = np.fromstring(model_res["fn"][0].strip('[]'), sep=' ')
    phin = np.fromstring(model_res["phin"][0].strip('[]'), sep=' ')
    gn = np.fromstring(model_res["gn"][0].strip('[]'), sep=' ')
    res_dict = {"fn": fn, "phin": phin, "gn": gn, "k": model_res["k"][0], "p": int(model_res["p"])}
    plot_metrics(res_dict)
