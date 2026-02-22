### Разминка

Найди гессиан функции

$$
f(x) = \|Ax - b\|^2, \quad x \in \mathbb{R}^n, 
$$

где $A$ — матрица размером $m \times n$, $b \in \mathbb{R}^m$.

Подставь значение

$$
A = \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}.
$$

В ответе укажи сумму элементов матрицы гессиана.

**Решение**

$$
f(x) = \|Ax - b\|^2 = (Ax - b)^T (Ax - b)
$$
$$
f(x) = x^T A^T A x - 2 b^T A x + b^T b
$$
$$
\nabla f(x) = 2A^T(Ax - b).
$$
$$
\nabla^2 f(x) = 2A^T A.
$$
$$
A = \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}
\quad

A^T =
\begin{pmatrix}
1 & 3 \\
2 & 4
\end{pmatrix}
$$

$$
A^T A =
\begin{pmatrix}
1 & 3 \\
2 & 4
\end{pmatrix}
\begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}
=
\begin{pmatrix}
10 & 14 \\
14 & 20
\end{pmatrix}
$$

$$
\nabla^2 f(x) = 2A^T A =
\begin{pmatrix}
20 & 28 \\
28 & 40
\end{pmatrix}
$$

$$
20 + 28 + 28 + 40 = 116
$$
