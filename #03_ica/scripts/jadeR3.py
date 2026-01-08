import numpy as np
from numpy import matrix, ndarray, arange, diag, eye, sqrt, zeros, concatenate
from numpy.linalg import eig, pinv

def jadeR(X, m=None, verbose=False):
    assert isinstance(X, ndarray), "X must be a numpy array"
    X = matrix(X.astype(np.float64))
    n, T = X.shape
    assert n < T, "n_sensors must be < n_samples"
    if m is None:
        m = n
    assert m <= n, "m must be <= n_sensors"
    X -= X.mean(1)

    # PCA for whitening
    D, U = eig((X * X.T) / float(T))
    idx = D.argsort()
    Ds = D[idx[-m:]]
    U_sub = U[:, idx[-m:]]
    scales = np.sqrt(Ds)
    B = diag(1.0 / scales) @ U_sub.T
    X_white = B @ X

    # Estimate cumulants
    Xq = X_white.T
    nbcm = int(m * (m + 1) / 2)
    CM = matrix(zeros([m, m*nbcm], dtype=np.float64))
    R = eye(m)
    Range = arange(m)
    for p in range(m):
        Xp = Xq[:, p]
        Xpp = np.multiply(Xp, Xp)
        Q = (np.multiply(Xpp, Xq).T @ Xq) / float(T) - R - 2 * (R[:, p] * R[:, p].T)
        CM[:, Range] = Q
        Range += m
        for q in range(p):
            Xpq = np.multiply(Xp, Xq[:, q])
            Q2 = np.sqrt(2) * ((np.multiply(Xpq, Xq).T @ Xq) / float(T)
                               - R[:, p] * R[:, q].T - R[:, q] * R[:, p].T)
            CM[:, Range] = Q2
            Range += m

    # Joint diagonalization
    V = eye(m)
    seuil = 1e-6 / sqrt(T)
    encore = True
    while encore:
        encore = False
        for p in range(m-1):
            for q in range(p+1, m):
                Ip = arange(p, m*nbcm, m)
                Iq = arange(q, m*nbcm, m)
                # Compute Givens rotation parameters
                g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                gg = g @ g.T
                ton = gg[0,0] - gg[1,1]
                toff = gg[0,1] + gg[1,0]
                theta = 0.5 * np.arctan2(toff, ton + np.sqrt(ton**2 + toff**2))
                if abs(theta) > seuil:
                    encore = True
                    c, s = np.cos(theta), np.sin(theta)
                    G = matrix([[c, -s], [s, c]])
                    pair = [p, q]
                    V[:, pair] = V[:, pair] @ G
                    CM[pair, :] = G.T @ CM[pair, :]
                    # Correctly update CM columns for Ip and Iq
                    updated_block = np.concatenate([
                        c*CM[:, Ip] + s*CM[:, Iq],
                        -s*CM[:, Ip] + c*CM[:, Iq]
                    ], axis=1)
                    CM[:, np.concatenate([Ip, Iq])] = updated_block

    # Final separation matrix
    B_sep = V.T @ B
    A_mat = pinv(B_sep)
    norms = np.sum(np.array(A_mat)**2, axis=0)
    order = np.argsort(-norms)
    B_sep = B_sep[order, :]
    return np.asarray(B_sep), np.asarray(U_sub), np.asarray(Ds), np.asarray(V)

def dewhiten_signals(X, B, U_sub, Ds, V):
    Xc = X - X.mean(axis=1, keepdims=True)
    Y_pca = U_sub.T @ Xc
    scales = np.sqrt(Ds)
    Y_white = Y_pca / scales[:, None]
    Y_ica = V.T @ Y_white
    S_orig = scales[:, None] * Y_ica
    variances = np.var(S_orig, axis=1)
    return S_orig, variances

if __name__ == "__main__":
    X = np.random.randn(5, 1000)
    B, U_sub, Ds, V = jadeR(X, m=5)
    S_orig, variances = dewhiten_signals(X, B, U_sub, Ds, V)
    print("Original-scale variances:", variances)
