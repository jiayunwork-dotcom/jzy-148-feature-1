"""逻辑回归，牛顿迭代（IRLS）自实现，只用 numpy。

设计矩阵第一列为常数 1（截距）。收敛判据：梯度最大绝对值 < tol。
带截距时梯度第一分量 = sum(p - y)，收敛即全样本平均预测概率等于样本违约率。

完全分离（数据被某个超平面完美切开）时逻辑回归 MLE 不存在、系数发散，
牛顿法的表现是：海森在饱和样本支撑集上逐渐奇异、系数沿固定方向线性增长、
单步长度不衰减（而正常收敛末期步长迅速趋零），同时概率在未充分饱和的
支撑样本上被持续推向极端。这里用「系数已经很大且最近一步仍把概率显著
推离当前值」判定发散，避免饱和后浮点梯度假性归零造成的误判为收敛。
海森确实奇异（无有效支撑）或达到最大次数同样判失败。
"""
from __future__ import annotations

import numpy as np

from .exceptions import BuildError

# 系数规模与单步概率推动超过这些值即认定为分离发散
BETA_DIVERGE = 20.0
PUSH_DIVERGE = 1e-3


def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _solve_hessian(H: np.ndarray, grad: np.ndarray):
    """解 H step = grad；H 奇异时加递减脊重试，仍失败则抛 BuildError。"""
    try:
        return np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        pass
    ridge = 1e-10
    for _ in range(6):
        try:
            Hr = H + ridge * np.eye(H.shape[0])
            return np.linalg.solve(Hr, grad)
        except np.linalg.LinAlgError:
            ridge *= 100
    raise BuildError("海森矩阵奇异且加正则后仍无法求解，牛顿迭代无法继续"
                     "（数据可能存在完全分离）")


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[np.ndarray, int, list[dict]]:
    """返回 (beta, 迭代次数, 迭代历史)。历史逐轮记录梯度最大分量、负对数似然、
    系数，作业失败时一并落库可查。不收敛时抛 BuildError。"""
    n, k = X.shape
    beta = np.zeros(k, dtype=np.float64)
    prev_nll = np.inf
    recent_steps: list[float] = []
    history: list[dict] = [
        {"iter": 0, "grad_max_abs": None, "neg_log_lik": None,
         "beta": [0.0] * k}
    ]

    for it in range(1, max_iter + 1):
        eta = np.clip(X @ beta, -700, 700)
        p = sigmoid(eta)
        grad = X.T @ (p - y) / n
        w = p * (1.0 - p)
        H = (X * w[:, None]).T @ X / n
        step = _solve_hessian(H, grad)

        beta_new = beta - step
        eta_new = np.clip(X @ beta_new, -700, 700)
        p_new = sigmoid(eta_new)
        nll = -np.mean(y * np.log(p_new + 1e-300)
                       + (1 - y) * np.log(1 - p_new + 1e-300))

        history.append({
            "iter": it,
            "grad_max_abs": float(np.max(np.abs(grad))),
            "neg_log_lik": float(nll) if np.isfinite(nll) else None,
            "step_max_abs": float(np.max(np.abs(step))),
            "beta": [float(v) for v in beta_new],
        })

        if not np.isfinite(nll) or np.max(np.abs(beta_new)) > 1e6:
            raise _fail(
                history,
                f"第 {it} 次迭代系数发散（|beta|>1e6 或对数似然非有限），"
                "数据可能存在完全分离，逻辑回归不收敛")

        grad_max = float(np.max(np.abs(grad)))
        step_max = float(np.max(np.abs(step)))

        # 分离发散检测：系数已经很大、梯度仍未达 tol，但最近 3 步的步长没有
        # 衰减（正常二次收敛步长应随梯度快速趋零；分离时步长稳定在一个常数，
        # 系数沿固定方向线性增长）。
        recent_steps.append(step_max)
        if len(recent_steps) > 3:
            recent_steps.pop(0)
        if (np.max(np.abs(beta_new)) > BETA_DIVERGE
                and grad_max > tol
                and len(recent_steps) == 3
                and min(recent_steps) >= 0.5 * max(recent_steps)):
            raise _fail(
                history,
                f"第 {it} 次迭代判定发散：|beta|={np.max(np.abs(beta_new)):.2f} "
                f"已很大，梯度 {grad_max:.2e} 未达 tol={tol:.0e}，而最近 3 步"
                f"步长不衰减（{recent_steps[0]:.3g}/{recent_steps[1]:.3g}/"
                f"{recent_steps[2]:.3g}），系数沿固定方向线性增长，"
                "数据存在完全分离，逻辑回归不收敛")

        # 真实收敛：梯度达到 tol。此时系数若异常大，上面的步长检测本应先触发；
        # 兜底再要求大系数情形下本步概率推动很小，避免饱和浮点假性归零。
        if grad_max < tol:
            push = float(np.max(np.abs(p_new - p)))
            if np.max(np.abs(beta_new)) > BETA_DIVERGE and push > PUSH_DIVERGE:
                raise _fail(
                    history,
                    f"第 {it} 次迭代梯度虽降至 {grad_max:.2e}，但 |beta|="
                    f"{np.max(np.abs(beta_new)):.2f} 已很大且本步仍把预测概率"
                    f"最大推动 {push:.3e}，判定为完全分离导致的发散，"
                    "逻辑回归不收敛")
            return beta_new, it, history

        if abs(prev_nll - nll) < tol and grad_max < tol * 100:
            return beta_new, it, history
        if it > 1 and nll > prev_nll + 1e-8:
            raise _fail(
                history,
                f"第 {it} 次迭代负对数似然不降反升（{prev_nll:.8f} -> "
                f"{nll:.8f}），牛顿迭代不收敛")
        beta = beta_new
        prev_nll = nll

    raise _fail(
        history,
        f"牛顿迭代达到最大次数 {max_iter} 仍未收敛（梯度最大分量 "
        f"{grad_max:.3e} > tol={tol:.0e}）")


def _fail(history: list[dict], message: str) -> BuildError:
    err = BuildError(message)
    err.history = history  # type: ignore[attr-defined]
    return err


def predict_proba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return sigmoid(np.clip(X @ beta, -700, 700))
