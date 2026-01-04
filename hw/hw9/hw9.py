import numpy as np

# -----------------------------
# 工具：漂亮檢查
# -----------------------------
def check_close(name, A, B, tol=1e-8):
    ok = np.allclose(A, B, atol=tol, rtol=tol)
    err = np.linalg.norm(A - B)
    print(f"[{name}] allclose={ok}, ||A-B||_F={err:.3e}")
    return ok

# ============================================================
# 1) 用遞迴（Laplace 展開）計算行列式（適合小矩陣）
# ============================================================
def det_recursive(A: np.ndarray) -> float:
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("det_recursive: A must be square")

    # base cases
    if n == 0:
        return 1.0
    if n == 1:
        return A[0, 0]
    if n == 2:
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]

    # Laplace expansion along first row (i = 0)
    det = 0.0
    for j in range(n):
        if A[0, j] == 0:
            continue
        minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
        det += ((-1) ** j) * A[0, j] * det_recursive(minor)
    return det

# ============================================================
# 2) 寫 LU 分解（含部分樞紐 pivoting），再用 LU 算 det
#    A = P^T L U   (我們用 row pivoting: P A = L U)
#    det(A) = det(P) * det(L) * det(U) = det(P) * prod(diag(U))
#    det(P) = (-1)^{#swaps}
# ============================================================
def lu_decomposition_partial_pivot(A: np.ndarray):
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    if n != m:
        raise ValueError("LU: A must be square")
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    swap_count = 0

    for k in range(n):
        # pivot row
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if np.isclose(U[pivot, k], 0.0):
            continue  # singular or near singular

        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
            swap_count += 1

        # elimination
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k] if not np.isclose(U[k, k], 0.0) else 0.0
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
            U[i, k] = 0.0

    return P, L, U, swap_count

def det_via_lu(A: np.ndarray) -> float:
    P, L, U, swap_count = lu_decomposition_partial_pivot(A)
    sign = -1.0 if (swap_count % 2 == 1) else 1.0
    return sign * np.prod(np.diag(U))

# ============================================================
# 3) 驗證：LU / Eigen / SVD 分解後能重建 A
# ============================================================
def verify_decompositions(A: np.ndarray):
    A = np.asarray(A, dtype=float)

    # LU: P A = L U  => A = P^T L U
    P, L, U, swaps = lu_decomposition_partial_pivot(A)
    A_lu_recon = P.T @ L @ U
    check_close("LU recon (A vs P^T L U)", A, A_lu_recon)

    # Eigen-decomposition (只對可對角化情況、且通常需方陣)
    # 對一般實矩陣，特徵向量可能為複數；這裡用 complex 以保險
    w, V = np.linalg.eig(A.astype(complex))
    D = np.diag(w)
    A_eig_recon = V @ D @ np.linalg.inv(V)
    check_close("Eigen recon (complex)", A.astype(complex), A_eig_recon)

    # SVD
    U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
    S = np.diag(s)
    A_svd_recon = U_svd @ S @ Vt
    check_close("SVD recon", A, A_svd_recon)

# ============================================================
# 4) 用特徵值分解做 SVD（透過 A^T A 或 A A^T）
#    A^T A = V (Sigma^2) V^T
#    U = A V Sigma^{-1}
# ============================================================
def svd_via_eigendecomposition(A: np.ndarray, tol=1e-12):
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    # 做 A^T A 的特徵值分解（對稱半正定）
    AtA = A.T @ A
    eigvals, V = np.linalg.eigh(AtA)  # eigh: for symmetric
    # eigvals ascending -> sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    # singular values
    s = np.sqrt(np.clip(eigvals, 0, None))

    # 計算 U
    U = np.zeros((m, len(s)))
    for i, si in enumerate(s):
        if si > tol:
            U[:, i] = (A @ V[:, i]) / si
        else:
            # 對應零奇異值：U 的該列可以任選正交補空間
            U[:, i] = 0.0

    # 讓 U 的有效部分做一次正交化（數值更穩）
    # 只對非零奇異值的向量做 QR
    nonzero = s > tol
    if np.any(nonzero):
        Q, _ = np.linalg.qr(U[:, nonzero])
        U[:, nonzero] = Q

    # 組回：A ≈ U Sigma V^T
    # 這裡回傳 full_matrices=False 的版本尺寸：U(m,k), Sigma(k,k), Vt(k,n)
    k = min(m, n)
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = V[:, :k].T
    return U_k, s_k, Vt_k

# ============================================================
# 5) PCA 主成分分析（用 SVD）
#    X: (n_samples, n_features)
#    回傳：主成分方向 components、投影 scores、解釋變異比 explained_variance_ratio
# ============================================================
def pca_svd(X: np.ndarray, n_components: int):
    X = np.asarray(X, dtype=float)
    # 1) center
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    # 2) SVD on centered data
    # Xc = U S V^T
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    # 3) components (principal axes)
    components = Vt[:n_components, :]  # (n_components, n_features)

    # 4) scores (projected coordinates)
    scores = Xc @ components.T  # (n_samples, n_components)

    # 5) explained variance
    n_samples = X.shape[0]
    eigenvalues = (s**2) / (n_samples - 1)  # covariance eigenvalues
    total_var = eigenvalues.sum()
    explained_variance_ratio = eigenvalues[:n_components] / total_var

    return {
        "mean": mean.squeeze(),
        "components": components,
        "scores": scores,
        "explained_variance_ratio": explained_variance_ratio,
    }

# ============================================================
# Demo / 測試
# ============================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # --- (1)(2) determinant ---
    A = np.array([
        [2.0, 1.0, 3.0],
        [4.0, 1.0, 6.0],
        [1.0, -1.0, 0.0]
    ])
    print("A=\n", A)

    det_r = det_recursive(A)
    det_lu = det_via_lu(A)
    det_np = np.linalg.det(A)

    print("\nDeterminant:")
    print("  det_recursive =", det_r)
    print("  det_via_lu    =", det_lu)
    print("  np.linalg.det =", det_np)

    # --- (3) verify decompositions ---
    print("\nVerify decompositions:")
    verify_decompositions(A)

    # --- (4) SVD via eigen ---
    print("\nSVD via eigendecomposition:")
    Ue, se, Vte = svd_via_eigendecomposition(A)
    A_recon = Ue @ np.diag(se) @ Vte
    check_close("SVD(eig) recon", A, A_recon)

    # compare to numpy svd
    Un, sn, Vtn = np.linalg.svd(A, full_matrices=False)
    # singular values may match (up to tiny numerical diffs)
    print("  singular values (eig) =", se)
    print("  singular values (np)  =", sn)

    # --- (5) PCA demo ---
    print("\nPCA demo:")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    # make correlated features
    X[:, 2] = 0.8 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * rng.normal(size=200)

    pca = pca_svd(X, n_components=2)
    print("  mean =", pca["mean"])
    print("  components=\n", pca["components"])
    print("  explained_variance_ratio =", pca["explained_variance_ratio"])
    print("  scores shape =", pca["scores"].shape)
