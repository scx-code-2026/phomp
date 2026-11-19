# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The PHOMP Authors
"""层常数生成器（SageMath 10.9 FreeLieAlgebra 版；Jacobi 自动约化）。

- 系数取在自由李代数中：叶 Gk，内部只有交换子 [X,Y]；
- Sage 自动做反对称与 Jacobi 化简；
- 直接输出论文的交换子形式。
用法：
    sage phomp_constants_sage.py 2 2
"""
import sys
from sage.all import QQ, factorial, bernoulli
from sage.algebras.lie_algebras.free_lie_algebra import FreeLieAlgebra


def zero():
    return [L.zero() for _ in range(D + 1)]


def add(x, y):
    return [a + b for a, b in zip(x, y)]


def sub(x, y):
    return [a - b for a, b in zip(x, y)]


def scale(x, c):
    return [a * c for a in x]


def comm(x, y):
    z = zero()
    for i, a in enumerate(x):
        for j, b in enumerate(y):
            if i + j <= D:
                z[i + j] += a.bracket(b)
    return z


def integral(x):
    z = zero()
    for k in range(D):
        z[k + 1] = x[k] / (k + 1)
    return z


def derivative(x):
    z = zero()
    for k in range(D):
        z[k] = (k + 1) * x[k + 1]
    return z


def dexp(S, Sp):
    out = zero()
    term = Sp
    for k in range(D + 1):
        if all(a.is_zero() for a in term):
            break
        out = add(out, scale(term, QQ(1) / factorial(k + 1)))
        term = comm(S, term)
    return out


def Ad(S, X):
    out = zero()
    term = X
    for k in range(D + 1):
        if all(a.is_zero() for a in term):
            break
        out = add(out, scale(term, QQ((-1) ** k) / factorial(k)))
        term = comm(S, term)
    return out


def solve_Omega(S, Dser):
    T = add(scale(S, 2), Dser)
    p_lead = next((k for k, a in enumerate(Dser) if not a.is_zero()), D)
    jmax = (D - p_lead) // (2 * (p_lead + 1))
    Om = zero()
    for _ in range(max(8, 2 * D)):
        rhs = sub(Dser, scale(comm(Om, T), QQ(1) / 2))
        for j in range(1, jmax + 1):
            term = Dser
            for _ in range(2 * j):
                term = comm(Om, term)
            rhs = add(rhs, scale(term, bernoulli(2 * j) / factorial(2 * j)))
        new = integral(rhs)
        if new == Om:
            return new
        Om = new
    print("WARNING: Picard iteration did not converge", file=sys.stderr)
    return Om


def one_layer(A, p, K):
    C = zero()
    for j in range(K):
        C[p + j] = A[p + j]
    Sigma = scale(integral(C), QQ(1) / 2)
    S = dexp(Sigma, derivative(Sigma))
    Dser = sub(Ad(Sigma, sub(A, S)), S)
    Om = solve_Omega(S, Dser)
    A_next = dexp(Om, derivative(Om))
    return A_next, min(2 * p + 2, p + K)


def main():
    global D, L
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    m_show = int(sys.argv[3]) if len(sys.argv) > 3 else None

    p = 0
    for _ in range(n):
        p = min(2 * p + 2, p + K)
    D = p + K
    L = FreeLieAlgebra(QQ, names=["G%d" % k for k in range(D + 1)])
    A = [L.gens()[k] for k in range(D + 1)]

    print("n=%d K=%d p_n=%d D=%d" % (n, K, p, D))
    p = 0
    for m in range(1, n + 1):
        A, p = one_layer(A, p, K)
        if m_show is None or m == m_show:
            print("\n层 %d: p_%d = %d" % (m, m, p))
            for j in range(K):
                if p + j <= D:
                    print("  A%d[%d] = %s" % (m, p + j, A[p + j]))
        if m_show is not None and m == m_show:
            break


if __name__ == "__main__":
    main()
