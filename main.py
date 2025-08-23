import numpy as np
from numpy import linalg as lng
import array
import time
from functools import reduce

# A vector is an array
# A matrix is also an array, but of more arrays (a collection of vectors)

# For example, if there are 3 vectors, A, B, C, then matrix1 is a 3x3 matrix consisting of those vectors

# Why arrays was used over the matrix class from numpy:
# Consistency: Aligns with standard NumPy practices used throughout the scientific Python ecosystem.
# Can represent vectors, matrices, and higher-dimensional tensors (perhaps applied later).
# NumPy functions applied to arrays consistently return arrays.
# NumPy development and new features are focused on arrays.

def checkmatssize(mh):
    row, col = mh[0].shape
    for mat in mh:
        if mat.shape != (row, col):
            print("Matrices are not of the same dimension.")
            time.sleep(1)
            print("Exiting program...")
            time.sleep(4)
            exit()
    return True

def addmats(mh):
    checkmatssize(mh)
    return np.sum(mh, axis=0).tolist()

def submats(mh):
    checkmatssize(mh)
    return reduce(np.subtract, mh).tolist()

def scalmultmat(mh, num):
    return [np.round(mat * num, 2).tolist() for mat in mh]

def multmats(mh):
    # Ensure at least 2 matrices
    if len(mh) < 2:
        return "Need at least 2 matrices to multiply."
    
    result = mh[0]
    for mat in mh[1:]:
        if result.shape[1] != mat.shape[0]:
            return "Matrix dimensions do not align for multiplication."
        result = np.matmul(result, mat)
    return result.tolist()

def transmat(mh):
    return [mat.T.tolist() for mat in mh]

def detmat(mh):
    return [round(lng.det(mat), 4) if mat.shape[0] == mat.shape[1] else "Not square" for mat in mh]

def invmat(mh):
    results = []
    for mat in mh:
        if mat.shape[0] != mat.shape[1]:
            results.append("Not square")
        elif np.isclose(lng.det(mat), 0):
            results.append("Singular matrix (no inverse)")
        else:
            results.append(np.round(lng.inv(mat), 4).tolist())
    return results

def cofactormat(mh):
    def cofactor(mat):
        n = mat.shape[0]
        cof = np.zeros_like(mat)
        for i in range(n):
            for j in range(n):
                minor = np.delete(np.delete(mat, i, axis=0), j, axis=1)
                cof[i, j] = ((-1)**(i+j)) * round(lng.det(minor), 4)
        return cof.tolist()
    
    result = []
    for mat in mh:
        if mat.shape[0] != mat.shape[1]:
            result.append("Not square")
        else:
            result.append(cofactor(mat))
    return result

def adjugatemat(mh):
    cof_mats = cofactormat(mh)
    result = []
    for cof in cof_mats:
        if isinstance(cof, str):
            result.append(cof)
        else:
            result.append(np.transpose(np.array(cof)).tolist())
    return result

def checkvecssize(vh):
    for i in range(len(vh)-1):
        if vh[i].size != vh[i+1].size:
            time.sleep(2)
            print("Vectors are different dimensions, hence cannot do this operation.")
            time.sleep(1)
            print("Exiting program...")
            time.sleep(4)
            exit()
    return True

def addvecs(vh):
    checkvecssize(vh)
    ans = np.array(vh)
    return np.sum(ans, axis=0)

def subvecs(vh):
    checkvecssize(vh)
    ans = np.array(vh)
    return reduce(np.subtract, ans)

def scalmult(vh, num):
    result = []
    for vector in vh:
        result.append(np.round(vector * num, 2).tolist())
    return result

def cp(vh):
    vh = [vh[0], vh[1]]
    checkvecssize(vh)
    if len(vh[0]) != 3:
        print("Cross product only works on vectors with 3 dimensions.")
        time.sleep(1)
        print("Exiting program...")
        time.sleep(4)
        exit()
    else:
        return np.cross(vh[0], vh[1]).tolist()

def dp(vh):
    checkvecssize(vh)
    return np.dot(vh[0], vh[1]).tolist()

def mag(vh):
    result = []
    for i in range(len(vh)):
        result.append(np.round(np.linalg.norm(vh[i]), 4))
    return result

def optype2(x, vh):
    if x in ("addition", "a", "add"):
        return addvecs(vh).tolist() if isinstance(vh[0], np.ndarray) and vh[0].ndim == 1 else addmats(vh)
    elif x in ("subtraction", "s", "sub"):
        return subvecs(vh).tolist() if isinstance(vh[0], np.ndarray) and vh[0].ndim == 1 else submats(vh)
    elif x in ("muliplication", "m", "scalar", "scalar multiplication", "sm"):
        scalar = float(input("Enter your scalar value_"))
        return scalmult(vh, scalar) if vh[0].ndim == 1 else scalmultmat(vh, scalar)
    elif x in ("matrix multiplication", "mm", "matmult", "matmul"):
        return multmats(vh)
    elif x in ("mag", "magnitude", "ma", "maggie", "maggy", "magnit", "magnitudes"):
        return mag(vh)
    elif x in ("dot product", "d", "dot", "d product", "dp", "sp", "scalar product"):
        return dp(vh)
    elif x in ("cross product", "c", "cross", "c product", "v", "cp", "vp"):
        return cp(vh)
    elif x in ("transpose", "t", "trans", "transpose matrix"):
        return transmat(vh)
    elif x in ("determinant", "deter", "det", "determinants"):
        return detmat(vh)
    elif x in ("inverse", "inv", "inverses"):
        return invmat(vh)
    elif x in ("cofactors", "cof", "cofactor matrix", "cofactor"):
        return cofactormat(vh)
    elif x in ("adjugate", "adj", "adjugates"):
        return adjugatemat(vh)
    else:
        return "Operation not recognised."

def optype1(x):
    if x in ("vector", "v", "vec"):
        varnumber = int(input("How many vectors?_"))  # how many vectors to operate on
        vectorsholder = []  # use a list to hold vectors
        for i in range(varnumber):
            vectorin = input(f"Enter the numbers for vector {i + 1} (no spaces): ")
            try:
                vectorout = np.array([int(char) for char in vectorin])
                vectorsholder.append(vectorout)  # append vector to the list
                print("Stored!")
            except ValueError:
                print(f"Invalid input for vector {i + 1}. Please enter only numbers without spaces.")
        print("So your vectors are:", vectorsholder)

        choptype2 = input("Addition, subtraction, magnitudes, multiplication, dot or cross product?_").lower()
        print(optype2(choptype2, vectorsholder))

    elif x in ("matrix", "m", "mat"):
        matnumber = int(input("How many matrices?_"))
        matricesholder = []
        for i in range(matnumber):
            print(f"\nMatrix {i+1}:")
            rownum = int(input(f"Enter number of rows for matrix {i+1}: "))
            colnum = int(input(f"Enter number of columns for matrix {i+1}: "))
            rows = []
            print("Enter each row as a continuous number (e.g., 123 for [1, 2, 3])")
            for r in range(rownum):
                while True:
                    rowinput = input(f"Row {r + 1}: ")
                    if len(rowinput) != colnum:
                        print(f"Expected {colnum} digits. Please re-enter.")
                    else:
                        try:
                            row = [int(ch) for ch in rowinput]
                            rows.append(row)
                            break
                        except ValueError:
                            print("Invalid input. Enter only digits.")
            matrix = np.array(rows)
            matricesholder.append(matrix)

        print("\nSo your matrices are:")
        for idx, mat in enumerate(matricesholder, 1):
            print(f"Matrix {idx}:\n{mat}")

        choptype2 = input("\nWhat matrix operation would you like to perform? (e.g., addition, subtraction, transpose, etc.)_ ").lower()
        print(optype2(choptype2, matricesholder))

    elif x in ("g", "guide", "gui"):
        print(
            '''
            Vectors:
            Addition: Adds each given vector together (A+B+C...)
            Subtraction: Subtracts each vector from the other sequentially (A-B-C)
            Magnitude: Finds the magnitude of each vector given
            Multiplication (Scalar): Multiplies the scalar onto each vector given (input scalar value later, enter vectors first)
            Dot Product: Scalar product of the (first) 2 vectors given
            Cross Product: Vector product of the (first) 2 vectors given

            Matrices:
            Addition: Adds each given matrix together (A+B+C)
            Subtraction: Subtracts each matrix from the other sequentially (A-B-C)
            Multiplication (Scalar): Multiplies the scalar onto each matrix given (input scalar value later)
            Matrix Multiplication: Multiplies matrices together in order (A*B*C) — requires matching dimensions
            Determinant: Finds the determinant of all given square matrices
            Transpose: Swaps rows and columns of all given matrices
            Cofactors Matrix: Finds the cofactor matrices of all given square matrices
            Adjugate: Finds the transpose of the cofactors matrix
            Inverse: Finds the inverse matrices of all given square, non-singular matrices
            '''
        )

if __name__ == "__main__":
    print(" \n")
    choptype1 = input("Vector or Matrix operation, or Guide (how this calculator works)?_").lower()
    optype1(choptype1)
