import numpy as np
from icspylab import sqrt_symmetric_matrix, sort_eigenvalues_eigenvectors
from icspylab import Scatter, cov, covW


def sort_IC_names(evals_M_star, lambda_ref):
    """
    Sort invariant components names in descending order of M_data eigenvalues.

    Parameters:
        evals_M_star (np.ndarray): Array of eigenvalues of M_star.
        lambda_ref (float): Corresponding eigenvectors.

    Returns:
        list: Sorted component names
    """
    IC_names = [f"IC_{i+1}" for i in range(evals_M_star.shape[0])]
    # Transform the eigenvalues of M_star to obtain those of M_data
    evals_Mdata_unsorted = (evals_M_star - lambda_ref)**2
    # Get the indices that would sort the M_data eigenvalues in descending order
    idx = evals_Mdata_unsorted.argsort()[::-1]
    # Sort the component names
    IC_names_sorted = [IC_names[i] for i in idx]
    return IC_names_sorted


def compute_M(X, lambda_ref, S1, S2, S1_args={}, S2_args={}):
    """
    Function to return the M matrix as in the paper, the centered data Xc, and cov_Xc_inv_sqrt = Cov(Xc)^{-0.5}.

    Parameters:
        X (array-like): data

    Returns:
        np.ndarray
    """

    n, p = X.shape

    # Compute Y: whitened data based on S1
    S1_X = S1(X, **S1_args)
    if not isinstance(S1_X, Scatter):
        raise ValueError("S1 must return a Scatter object")
    S1_X_inv_sqrt = sqrt_symmetric_matrix(S1_X.scatter, inverse=True)
    if S1_X.location is not None:
        Xc = X - S1_X.location
    else:
        Xc = X - np.mean(X, axis=0)
    Y = Xc @ S1_X_inv_sqrt

    # Compute M_star = S2(Y)
    S2_Y = S2(Y, **S2_args)
    if not isinstance(S2_Y, Scatter):
        raise ValueError("S2 must return an Scatter object")
    M_star = S2_Y.scatter

    # Compute M
    M = (M_star - lambda_ref * np.eye(p)) @ (M_star - lambda_ref * np.eye(p)).T

    return M


def compute_fi(eigvecs_boot, evecs_Mdata, max_comp):
    """
    Compute 1 - |det(U_k.T @ Ub_k)| where U_k and Ub_k are the first k eigenvectors of eigvecs_boot and evecs_Mdata, for
    k = 1, ..., max_comp.

    Parameters:
        eigvecs_boot (np.ndarray): matrix containing the eigenvectors of M(Xboot)
        evecs_Mdata (np.ndarray): matrix containing the eigenvectors of M(X)
        max_comp (int): number of components to consider

    Returns:
        np.ndarray (max_comp-1,)
    """
    fni = np.zeros(max_comp)
    for i in range(max_comp):
        mat = evecs_Mdata[:, :i+1].T @ eigvecs_boot[:, :i + 1]
        fni[i] = np.linalg.det(mat)
    return 1 - np.abs(fni)


def ladle_boot_iter(X, evecs_Mdata, max_comp, lambda_ref, S1, S2, S1_args, S2_args):
    """ Function for the bootstrap step. """
    Mboot = compute_M(X, lambda_ref, S1, S2, S1_args, S2_args)
    eigvals_boot, eigvecs_boot = np.linalg.eigh(Mboot)
    _, eigvecs_boot = sort_eigenvalues_eigenvectors(eigvals_boot, eigvecs_boot)
    return compute_fi(eigvecs_boot, evecs_Mdata, max_comp)


def ladle(X, S1=cov, S2=covW, S1_args={}, S2_args={}, lambda_ref=None, n_boot=200, max_comp=None, random_state=None):
    """
    Main ladle function

    Parameters:
        X (array-like): data
        S1 (function returning a scatter object): (default: cov) Function to compute the first scatter matrix.
        S2 (function returning a scatter object): (default: covW) Function to compute the second scatter matrix.
        S1_args (dict): Additional arguments for S1.
        S2_args (dict): Additional arguments for S2.
        lambda_ref (float or None): (default: None) Theoretical eigenvalue associated to the noise space. If None, then the median eigenvalue will be used as the reference.
        n_boot (int): (default: 200) number of bootstrapping samples to be taken
        max_comp (None, int): (default: None) number of components to consider. If None, it is computed in the function.
        random_state (None, int): (default: None) A seed to initialize np.random.default_rng. If None, then fresh, unpredictable entropy
        will be pulled from the OS.

    Returns:
        dict
    """
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    if max_comp is None:
        max_comp = p - 1 if p <= 10 else int(np.floor(p / np.log(p)))

    # Compute Y: whitened data based on S1
    S1_X = S1(X, **S1_args)
    if not isinstance(S1_X, Scatter):
        raise ValueError("S1 must return a Scatter object")
    S1_X_inv_sqrt = sqrt_symmetric_matrix(S1_X.scatter, inverse=True)
    if S1_X.location is not None:
        Xc = X - S1_X.location
    else:
        Xc = X - np.mean(X, axis=0)
    Y = Xc @ S1_X_inv_sqrt

    # Compute M_star = S2(Y)
    S2_Y = S2(Y, **S2_args)
    if not isinstance(S2_Y, Scatter):
        raise ValueError("S2 must return an Scatter object")
    M_star = S2_Y.scatter

    # Compute lambda_ref
    evals_M_star, evecs_M_star = np.linalg.eigh(M_star)
    evals_M_star, _ = sort_eigenvalues_eigenvectors(evals_M_star, evecs_M_star)

    if lambda_ref is None:
        lambda_ref = np.median(evals_M_star)
    print(f'lambda_ref = {lambda_ref}')

    # Sort component names
    IC_names_sorted = sort_IC_names(evals_M_star, lambda_ref)

    # Compute M(X)
    Mdata = (M_star - lambda_ref * np.eye(p)) @ (M_star - lambda_ref * np.eye(p)).T

    # Eigendecomposition of Mdata
    evals_Mdata, evecs_Mdata = np.linalg.eigh(Mdata)
    evals_Mdata, evecs_Mdata = sort_eigenvalues_eigenvectors(evals_Mdata, evecs_Mdata)

    # Bootstrap replicates to measure the variation of the eigenvectors
    fis = np.zeros(max_comp,)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        Xboot = X[idx, :]
        fis += ladle_boot_iter(Xboot, evecs_Mdata, max_comp, lambda_ref, S1, S2, S1_args, S2_args)
    fis /= n_boot

    # Compute fn: a measure of the variation of the eigenvectors
    fn0 = np.concatenate(([0], fis))
    fn = fn0 / (1 + np.sum(fn0))

    # Compute phin: the information provided by the eigenvalues (1 or lambda_med**2)
    phin = evals_Mdata[:max_comp + 1] / (1 + np.sum(evals_Mdata[:max_comp + 1]))

    # The ladle estimator of the number of components
    gn = fn + phin
    n_comp = np.argmin(gn)

    # Unmixing matrix W and final transformed matrix Z
    # W = evecs_Mdata.T @ S1_X_inv_sqrt
    # Z = Xc @ W.T

    RES = {"method": "FOBI", "n_comp": n_comp, "fn": fn, "phin": phin, "gn": gn,
           "lambda": evals_Mdata[:max_comp + 1], "IC_ordered": IC_names_sorted[:max_comp + 1]
           }

    return RES
