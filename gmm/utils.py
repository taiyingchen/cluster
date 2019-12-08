def read_input(filename):
    with open(filename, 'r') as file:
        N, k = file.readline().strip().split()
        N, k = int(N), int(k)
        X = []
        clusters = []

        for _ in range(N):
            buffers = file.readline().strip().split()
            X.append([float(v) for v in buffers])

        for _ in range(k):
            buffers = file.readline().strip().split()
            clusters.append([float(v) for v in buffers])

    return X, clusters


def add_2d_matrix(A, B):
    a1, b1, c1, d1 = A[0][0], A[0][1], A[1][0], A[1][1]
    a2, b2, c2, d2 = B[0][0], B[0][1], B[1][0], B[1][1]
    return [[a1+a2, b1+b2], [c1+c2, d1+d2]]


def det_2d_matrix(A):
    """Determinant of a 2d matrix
    A = [[a, b], [c, d]]
    det(A) = ad - bc
    """
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    return a * d - b * c


def invert_2d_matrix(A):
    """Inverse of a 2d matrix
    A = [[a, b], [c, d]]
    invert(A) = 1 / det(A) * [[d, -b], [-c, a]]
    """
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    scalar = 1 / det_2d_matrix(A)
    return [[scalar*d, scalar*-b], [scalar*-c, scalar*a]]


def mul_vector_2d_matrix(vector, A):
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    return [vector[0]*a + vector[1]*c, vector[0]*b + vector[1]*d]


def mul_scalar_2d_matrix(scalar, A):
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    return [[scalar*a, scalar*b], [scalar*c, scalar*d]]


def div_scalar_2d_matrix(scalar, A):
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    return [[a/scalar, b/scalar], [c/scalar, d/scalar]]


def add_vector(vector1, vector2):
    return [v1 + v2 for v1, v2 in zip(vector1, vector2)]


def sub_vector(vector1, vector2):
    return [v1 - v2 for v1, v2 in zip(vector1, vector2)]


def dot_vector(vector1, vector2):
    return sum([v1 * v2 for v1, v2 in zip(vector1, vector2)])


def mul_vector(vector1, vector2):
    mul = [[0. for j in range(len(vector1))] for i in range(len(vector1))]
    for i in range(len(vector1)):
        for j in range(len(vector2)):
            mul[i][j] = vector1[i] * vector2[j]
    return mul
