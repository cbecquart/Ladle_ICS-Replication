import numpy as np
from icspylab import cov, cov4, mcd, tM, tcov

N_SIMU = 500

LIST_NBOOT = [100, 200, 500]

dict_GMM1 = {
    "index": "GMM1",
    "model_label": "GMM",
    "eps": [0.15, 0.85],
    "mu": [
        np.array([1, 10, 0, 0, 0, 0]),
        np.array([10, 1, 0, 0, 0, 0]),
    ],
    "sigma": [np.eye(6) for _ in range(2)],
}

dict_GMM2 = {
    "index": "GMM2",
    "model_label": "GMM",
    "eps": [0.3, 0.3, 0.4],
    "mu": [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([0, 6, 0, 0, 0, 0]),
        np.array([10, 4, 0, 0, 0, 0]),
    ],
    "sigma": [np.eye(6) for _ in range(3)],
}

dict_GMM3 = {
    "index": "GMM3",
    "model_label": "GMM",
    "eps": [0.05, 0.1, 0.85],
    "mu": [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([0, 6, 0, 0, 0, 0]),
        np.array([10, 4, 0, 0, 0, 0]),
    ],
    "sigma": [np.eye(6) for _ in range(3)],
}

dict_GMM4 = {
    "index": "GMM4",
    "model_label": "GMM",
    "eps": [0.1, 0.45, 0.45],
    "mu": [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([0, 6, 0, 0, 0, 0]),
        np.array([10, 4, 0, 0, 0, 0]),
    ],
    "sigma": [np.eye(6) for _ in range(3)],
}

dict_GMM5 = {
    "index": "GMM5",
    "model_label": "GMM",
    "eps": [0.25, 0.25, 0.25, 0.25],
    "mu": [
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([0, 0, 10, 0, 0, 0, 0, 0, 0]),
        np.array([0, 10, 10, 0, 0, 0, 0, 0, 0]),
        np.array([10, 10, 10, 0, 0, 0, 0, 0, 0]),
    ],
    "sigma": [np.eye(9) for _ in range(4)],
}

dict_GMM6 = {
    "index": "GMM6",
    "model_label": "GMM",
    "eps": [0.03, 0.12, 0.35, 0.5],
    "mu": [
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([0, 0, 10, 0, 0, 0, 0, 0, 0]),
        np.array([0, 10, 10, 0, 0, 0, 0, 0, 0]),
        np.array([10, 10, 10, 0, 0, 0, 0, 0, 0]),
    ],
    "sigma": [np.eye(9) for _ in range(4)],
}

dict_GMM7 = {
    "index": "GMM7",
    "model_label": "GMM",
    "eps": [0.05, 0.05, 0.1, 0.8],
    "mu": [
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        np.array([0, 0, 10, 0, 0, 0, 0, 0, 0]),
        np.array([0, 10, 10, 0, 0, 0, 0, 0, 0]),
        np.array([10, 10, 10, 0, 0, 0, 0, 0, 0]),
    ],
    "sigma": [np.eye(9) for _ in range(4)],
}

dict_GMM8 = {
    "index": "GMM8",
    "model_label": "GMM",
    "eps": [0.45, 0.15, 0.4],
    "mu": [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([0, 6.5, 0, 0, 0, 0]),
        np.array([3, 4.5, 0, 0, 0, 0]),
    ],
    "sigma": [np.eye(6) for _ in range(3)],
}

dict_GMM9 = {
    "index": "GMM9",
    "model_label": "GMM",
    "eps": [0.4, 0.25, 0.35],
    "mu": [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([0, 6.5, 0, 0, 0, 0]),
        np.array([2.5, 4, 0, 0, 0, 0]),
    ],
    "sigma": [np.eye(6) for _ in range(3)],
}

LIST_MODELS = [
    dict_GMM2, dict_GMM3, dict_GMM4, dict_GMM5, dict_GMM6, dict_GMM7, dict_GMM8, dict_GMM9
]

dict_cov_cov4 = {"label": "cov_cov4","S1": cov, "S2": cov4, "S1_args": {}, "S2_args": {}}
dict_mcd05_cov = {"label": "mcd05_cov", "S1": mcd, "S2": cov, "S1_args": {"reweighted": False, "support_fraction": 0.5}, "S2_args": {}}
dict_mcd025_cov = {"label": "mcd025_cov", "S1": mcd, "S2": cov, "S1_args": {"reweighted": False, "support_fraction": 0.25}, "S2_args": {}}
dict_tM_cov = {"label": "tM_cov", "S1": tM, "S2": cov, "S1_args": {}, "S2_args": {}}
dict_tcov_cov = {"label": "tcov_cov", "S1": tcov, "S2": cov, "S1_args": {}, "S2_args": {}}

LIST_SCATTERS = [dict_cov_cov4, dict_mcd025_cov, dict_tM_cov]
