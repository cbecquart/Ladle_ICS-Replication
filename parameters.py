import numpy as np
from icspylab import cov, cov4, mcd, tM, tcov

N_SIMU = 500

LIST_NBOOT = [100, 200, 500]

dict_GMM1 = {
    "index": "GMM1",
    "model_label": "GMM",
    "eps": [0.15, 0.85],
    "mu": [np.array([1, 10]), np.array([10, 1])],
    "sigma": [np.eye(2) for _ in range(2)],
    "p": 6
}

dict_GMM2 = {
    "index": "GMM2",
    "model_label": "GMM",
    "eps": [0.3, 0.3, 0.4],
    "mu": [np.array([0, 0]), np.array([0, 6]), np.array([10, 4])],
    "sigma": [np.eye(2) for _ in range(3)],
    "p": 6
}

dict_GMM3 = {
    "index": "GMM3",
    "model_label": "GMM",
    "eps": [0.05, 0.1, 0.85],
    "mu": [np.array([0, 0]), np.array([0, 6]), np.array([10, 4])],
    "sigma": [np.eye(2) for _ in range(3)],
    "p": 6
}

dict_GMM4 = {
    "index": "GMM4",
    "model_label": "GMM",
    "eps": [0.1, 0.45, 0.45],
    "mu": [np.array([0, 0]), np.array([0, 6]), np.array([10, 4])],
    "sigma": [np.eye(2) for _ in range(3)],
    "p": 6
}

dict_GMM5 = {
    "index": "GMM5",
    "model_label": "GMM",
    "eps": [0.25, 0.25, 0.25, 0.25],
    "mu": [np.array([0, 0, 0]), np.array([0, 0, 10]), np.array([0, 10, 10]), np.array([10, 10, 10])],
    "sigma": [np.eye(3) for _ in range(4)],
    "p": 9
}

dict_GMM6 = {
    "index": "GMM6",
    "model_label": "GMM",
    "eps": [0.03, 0.12, 0.35, 0.5],
    "mu": [np.array([0, 0, 0]), np.array([0, 0, 10]), np.array([0, 10, 10]), np.array([10, 10, 10])],
    "sigma": [np.eye(3) for _ in range(4)],
    "p": 9
}

dict_GMM7 = {
    "index": "GMM7",
    "model_label": "GMM",
    "eps": [0.05, 0.05, 0.1, 0.8],
    "mu": [np.array([0, 0, 0]), np.array([0, 0, 10]), np.array([0, 10, 10]), np.array([10, 10, 10])],
    "sigma": [np.eye(3) for _ in range(4)],
    "p": 9
}

dict_GMM8 = {
    "index": "GMM8",
    "model_label": "GMM",
    "eps": [0.45, 0.15, 0.4],
    "mu": [np.array([0, 0]), np.array([0, 6.5]), np.array([3, 4.5])],
    "sigma": [np.eye(2) for _ in range(3)],
    "p": 6
}

dict_GMM9 = {
    "index": "GMM9",
    "model_label": "GMM",
    "eps": [0.4, 0.25, 0.35],
    "mu": [np.array([0, 0]), np.array([0, 6.5]), np.array([2.5, 4])],
    "sigma": [np.eye(2) for _ in range(3)],
    "p": 6
}

dict_SMM1_df3 = {
    "index": "SMM1_df3",
    "model_label": "SMM",
    "eps": dict_GMM1["eps"], "mu": dict_GMM1["mu"], "sigma": dict_GMM1["sigma"], "df": 3, "p": dict_GMM1["p"]
}

dict_SMM2_df3 = {
    "index": "SMM2_df3",
    "model_label": "SMM",
    "eps": dict_GMM2["eps"], "mu": dict_GMM2["mu"], "sigma": dict_GMM2["sigma"], "df": 3, "p": dict_GMM2["p"]
}

dict_SMM3_df3 = {
    "index": "SMM3_df3",
    "model_label": "SMM",
    "eps": dict_GMM3["eps"], "mu": dict_GMM3["mu"], "sigma": dict_GMM3["sigma"], "df": 3, "p": dict_GMM3["p"]
}

dict_SMM4_df3 = {
    "index": "SMM4_df3",
    "model_label": "SMM",
    "eps": dict_GMM4["eps"], "mu": dict_GMM4["mu"], "sigma": dict_GMM4["sigma"], "df": 3, "p": dict_GMM4["p"]
}

dict_SMM5_df3 = {
    "index": "SMM5_df3",
    "model_label": "SMM",
    "eps": dict_GMM5["eps"], "mu": dict_GMM5["mu"], "sigma": dict_GMM5["sigma"], "df": 3, "p": dict_GMM5["p"]
}

dict_SMM6_df3 = {
    "index": "SMM6_df3",
    "model_label": "SMM",
    "eps": dict_GMM6["eps"], "mu": dict_GMM6["mu"], "sigma": dict_GMM6["sigma"], "df": 3, "p": dict_GMM6["p"]
}

dict_SMM7_df3 = {
    "index": "SMM7_df3",
    "model_label": "SMM",
    "eps": dict_GMM7["eps"], "mu": dict_GMM7["mu"], "sigma": dict_GMM7["sigma"], "df": 3, "p": dict_GMM7["p"]
}

dict_SMM8_df3 = {
    "index": "SMM8_df3",
    "model_label": "SMM",
    "eps": [0.45, 0.55],
    "mu": [np.array([1, 10]), np.array([10, 1])],
    "sigma": [np.eye(2) for _ in range(2)],
    "df": 3,
    "p": 6
}


dict_PEM1 = {
    "index": "PEM1",
    "model_label": "PEM",
    "eps": dict_GMM1["eps"], "mu": dict_GMM1["mu"], "sigma": dict_GMM1["sigma"], "beta": 0.8, "p": dict_GMM1["p"]
}

dict_PEM2 = {
    "index": "PEM2",
    "model_label": "PEM",
    "eps": dict_GMM2["eps"], "mu": dict_GMM2["mu"], "sigma": dict_GMM2["sigma"], "beta": 0.8, "p": dict_GMM2["p"]
}

dict_PEM3 = {
    "index": "PEM3",
    "model_label": "PEM",
    "eps": dict_GMM3["eps"], "mu": dict_GMM3["mu"], "sigma": dict_GMM3["sigma"], "beta": 0.8, "p": dict_GMM3["p"]
}

dict_PEM4 = {
    "index": "PEM4",
    "model_label": "PEM",
    "eps": dict_GMM4["eps"], "mu": dict_GMM4["mu"], "sigma": dict_GMM4["sigma"], "beta": 0.8, "p": dict_GMM4["p"]
}

dict_PEM5 = {
    "index": "PEM5",
    "model_label": "PEM",
    "eps": dict_GMM5["eps"], "mu": dict_GMM5["mu"], "sigma": dict_GMM5["sigma"], "beta": 0.8, "p": dict_GMM5["p"]
}

dict_PEM6 = {
    "index": "PEM6",
    "model_label": "PEM",
    "eps": dict_GMM6["eps"], "mu": dict_GMM6["mu"], "sigma": dict_GMM6["sigma"], "beta": 0.8, "p": dict_GMM6["p"]
}

dict_PEM7 = {
    "index": "PEM7",
    "model_label": "PEM",
    "eps": dict_GMM7["eps"], "mu": dict_GMM7["mu"], "sigma": dict_GMM7["sigma"], "beta": 0.8, "p": dict_GMM7["p"]
}

dict_PEM8 = {
    "index": "PEM8",
    "model_label": "PEM",
    "eps": dict_GMM8["eps"], "mu": dict_GMM8["mu"], "sigma": dict_GMM8["sigma"], "beta": 0.8, "p": dict_GMM8["p"]
}

dict_PEM9 = {
    "index": "PEM9",
    "model_label": "PEM",
    "eps": dict_GMM9["eps"], "mu": dict_GMM9["mu"], "sigma": dict_GMM9["sigma"], "beta": 0.8, "p": dict_GMM9["p"]
}

dict_PEM10 = {
    "index": "PEM10",
    "model_label": "PEM",
    "eps": dict_GMM8["eps"], "mu": dict_GMM8["mu"], "sigma": dict_GMM8["sigma"], "beta": 1.2, "p": dict_GMM8["p"]
}

dict_PEM11 = {
    "index": "PEM11",
    "model_label": "PEM",
    "eps": dict_GMM9["eps"], "mu": dict_GMM9["mu"], "sigma": dict_GMM9["sigma"], "beta": 1.2, "p": dict_GMM9["p"]
}

LIST_MODELS = [
    dict_GMM1, dict_GMM2, dict_GMM3, dict_GMM4, dict_GMM5, dict_GMM6, dict_GMM7, dict_GMM8, dict_GMM9,
    dict_SMM1_df3, dict_SMM2_df3, dict_SMM3_df3, dict_SMM4_df3, dict_SMM5_df3, dict_SMM6_df3, dict_SMM7_df3, dict_SMM8_df3,
    dict_PEM1, dict_PEM2, dict_PEM3, dict_PEM4, dict_PEM5, dict_PEM6, dict_PEM7, dict_PEM8, dict_PEM9, dict_PEM10, dict_PEM11
]

dict_cov_cov4 = {"label": "cov_cov4","S1": cov, "S2": cov4, "S1_args": {}, "S2_args": {}}
dict_mcd05_cov = {"label": "mcd05_cov", "S1": mcd, "S2": cov, "S1_args": {"reweighted": False, "support_fraction": 0.5}, "S2_args": {}}
dict_mcd025_cov = {"label": "mcd025_cov", "S1": mcd, "S2": cov, "S1_args": {"reweighted": False, "support_fraction": 0.25}, "S2_args": {}}
dict_tM_cov = {"label": "tM_cov", "S1": tM, "S2": cov, "S1_args": {}, "S2_args": {}}
dict_tcov_cov = {"label": "tcov_cov", "S1": tcov, "S2": cov, "S1_args": {}, "S2_args": {}}

LIST_SCATTERS = [dict_cov_cov4, dict_mcd05_cov, dict_mcd025_cov, dict_tM_cov, dict_tcov_cov]
