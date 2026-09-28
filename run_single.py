import argparse
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from ladle import ladle
from icspylab import ICS, cov, cov4, mcd, tM, tcov
from icspylab.distributions import (
    generate_gaussian_mixture,
    generate_student_mixture,
    generate_powerexp_mixture
)

from parameters import N_SIMU, LIST_MODELS, LIST_SCATTERS


N_DATA = 5000
N_BOOT = 200

TOTAL_TASKS = len(LIST_MODELS) * N_SIMU


def generate_data(dict_model):

    model_label = dict_model["model_label"]

    if model_label == "GMM":
        return generate_gaussian_mixture(
            eps=dict_model["eps"],
            mu=dict_model["mu"],
            sigma=dict_model["sigma"],
            n=N_DATA,
            p=dict_model["p"],
        )

    elif model_label == "SMM":
        return generate_student_mixture(
            eps=dict_model["eps"],
            mu=dict_model["mu"],
            sigma=dict_model["sigma"],
            df=dict_model["df"],
            n=N_DATA,
            p=dict_model["p"],
        )

    elif model_label == "PEM":
        return generate_powerexp_mixture(
            eps=dict_model["eps"],
            mu=dict_model["mu"],
            sigma=dict_model["sigma"],
            beta=dict_model["beta"],
            n=N_DATA,
            p=dict_model["p"],
        )

    else:
        raise ValueError("Unknown model")


def dftu(X, S1, S2, S1_args={}, S2_args={}):

    ics_dftu = ICS(S1=S1, S2=S2, S1_args=S1_args, S2_args=S2_args, algorithm="whiten", method_select="unimodal")
    ics_dftu.fit(X)

    return ics_dftu


def get_dict_res(res, res_ftu, dict_scatter, dict_model, args, real_id, simu_idx):
    dict_res = {
        "m": args.m,
        "task_id": args.task_id,
        "real_id": real_id,
        "simu_idx": simu_idx,
        "model_index": dict_model["index"],
        "p": dict_model["p"],
        "k": len(dict_model["eps"]),
        "model_label": dict_model["model_label"],
        "model_desc": f"eps={dict_model["eps"]}, mu={dict_model["mu"]}, sigma={dict_model["sigma"]}",
        "scatters": dict_scatter["label"],
        "S1": dict_scatter["S1"].__name__, "S2": dict_scatter["S2"].__name__,
        "S1_args": dict_scatter["S1_args"], "S2_args": dict_scatter["S2_args"],
        "n_boot": N_BOOT,
        "fn": res["fn"],
        "lambda": res["lambda"],
        "phin": res["phin"],
        "gn": res["gn"],
        "n_comp": res["n_comp"],
        "IC_ordered": res["IC_ordered"],
        "dftu_ncomp": res_ftu.n_components_,
        "dftu_comp": res_ftu.component_names_
    }
    return dict_res


def get_dict_res_cov_cov4(res, dict_scatter, dict_model, args, real_id, simu_idx):
    dict_res = {
        "m": args.m,
        "task_id": args.task_id,
        "real_id": real_id,
        "simu_idx": simu_idx,
        "model_index": dict_model["index"],
        "p": dict_model["p"],
        "k": len(dict_model["eps"]),
        "model_label": dict_model["model_label"],
        "model_desc": f"eps={dict_model["eps"]}, mu={dict_model["mu"]}, sigma={dict_model["sigma"]}",
        "scatters": "cov_cov4_ref",
        "S1": dict_scatter["S1"].__name__, "S2": dict_scatter["S2"].__name__,
        "S1_args": dict_scatter["S1_args"], "S2_args": dict_scatter["S2_args"],
        "n_boot": N_BOOT,
        "fn": res["fn"],
        "lambda": res["lambda"],
        "phin": res["phin"],
        "gn": res["gn"],
        "n_comp": res["n_comp"],
        "IC_ordered": res["IC_ordered"],
        "dftu_ncomp": None,
        "dftu_comp": None
    }

    return dict_res


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, required=True, help='ID multiplier')
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Use task_id as seed base for reproducibility across runs
    real_id = 5 * args.m + args.task_id
    seed = 42 + real_id * 42
    np.random.seed(seed)
    if real_id >= TOTAL_TASKS:
        raise ValueError(f"real_id={real_id} exceeds TOTAL_TASKS={TOTAL_TASKS}")

    model_idx = real_id % len(LIST_MODELS)
    simu_idx = real_id // len(LIST_MODELS)

    dict_model = LIST_MODELS[model_idx]

    data, _ = generate_data(dict_model)
    results = []

    for dict_scatter in LIST_SCATTERS:
        res = ladle(
            data,
            S1=dict_scatter["S1"],
            S2=dict_scatter["S2"],
            S1_args=dict_scatter["S1_args"],
            S2_args=dict_scatter["S2_args"],
            n_boot=N_BOOT,
            lambda_ref=None
        )

        res_ftu = dftu(
            data,
            S1=dict_scatter["S1"],
            S2=dict_scatter["S2"],
            S1_args=dict_scatter["S1_args"],
            S2_args=dict_scatter["S2_args"],
        )

        dict_res = get_dict_res(res=res, res_ftu=res_ftu, dict_scatter=dict_scatter, dict_model=dict_model,
                                args=args, real_id=real_id, simu_idx=simu_idx)
        results.append(dict_res)

        if dict_scatter["label"] == "cov_cov4":
            res = ladle(
                data,
                S1=dict_scatter["S1"],
                S2=dict_scatter["S2"],
                S1_args=dict_scatter["S1_args"],
                S2_args=dict_scatter["S2_args"],
                n_boot=N_BOOT,
                lambda_ref=1
            )

            dict_res = get_dict_res_cov_cov4(res=res, dict_scatter=dict_scatter, dict_model=dict_model,
                                             args=args, real_id=real_id, simu_idx=simu_idx)
            results.append(dict_res)

    df = pd.DataFrame(results)

    out_path = os.path.join(
        args.output_dir,
        f"result_{real_id}.csv"
    )

    df.to_csv(out_path, index=False)

    print(f"Saved {out_path}")


if __name__ == "__main__":
    start = timer()
    main()

    elapsed = int(timer() - start)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60
    print(f"Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
