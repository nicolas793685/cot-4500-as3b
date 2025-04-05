import numpy as np

# Question 1: Gaussian Elimination with Back Substitution
def gaussian_elimination_solve(aug_matrix):
    A = aug_matrix.astype(float)
    n = len(A)
    
    # Forward elimination
    for i in range(n):
        for j in range(i+1, n):
            if A[i, i] == 0:
                raise ValueError("Zero pivot encountered!")
            ratio = A[j, i] / A[i, i]
            A[j, i:] -= ratio * A[i, i:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (A[i, -1] - np.dot(A[i, i+1:n], x[i+1:n])) / A[i, i]
    return x

# Question 2: LU Decomposition
def lu_factorization(matrix):
    n = matrix.shape[0]
    L = np.zeros_like(matrix, dtype=float)
    U = np.zeros_like(matrix, dtype=float)
    A = matrix.astype(float)

    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i+1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    determinant = np.linalg.det(matrix)
    return L, U, determinant

# Question 3: Diagonal Dominance
def is_diagonally_dominant(matrix):
    for i in range(len(matrix)):
        row_sum = sum(abs(matrix[i][j]) for j in range(len(matrix)) if j != i)
        if abs(matrix[i][i]) < row_sum:
            return False
    return True

# Question 4: Positive Definiteness
def is_positive_definite(matrix):
    if not np.allclose(matrix, matrix.T):
        return False
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

# Running Question 1
a1 = np.array([
    [2, -1, 1, 6],
    [1,  3, 1, 0],
    [-1, 5, 4, -3]
])
solution1 = gaussian_elimination_solve(a1)
print("Solution to Question 1 (Gaussian Elimination):", solution1)

# Running Question 2
matrix2 = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
])
L, U, det = lu_factorization(matrix2)
print("\nQuestion 2 Results:")
print("Determinant: {:.14f}".format(det))
print("L matrix:\n", L)
print("U matrix:\n", U)

# Running Question 3
matrix3 = np.array([
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
])
is_diag_dom = is_diagonally_dominant(matrix3)
print("\nQuestion 3: Is the matrix diagonally dominant?", is_diag_dom)

# Running Question 4
matrix4 = np.array([
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2]
])
is_pos_def = is_positive_definite(matrix4)
print("\nQuestion 4: Is the matrix positive definite?", is_pos_def)
