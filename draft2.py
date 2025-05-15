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

def checkvecssize(vh):
    for i in range(len(vh)-1):
        if vh[i].size != vh[i+1].size:
            time.sleep(2)
            print("Vectors are different dimensions, hence cannont do this operation.")
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
        return addvecs(vh).tolist()
    elif x in ("subtraction", "s", "sub"):
         return subvecs(vh).tolist()
    elif x in ("muliplication", "m", "mult", "scalar", "scalar multiplication", "sm"):
        scalar = float(input("Enter your scalar value_"))
        return scalmult(vh, scalar)
    elif x in ("mag", "magnitude", "ma", "maggie", "maggy", "magnit", "magnitudes"):
        return mag(vh)
    elif x in ("dot product", "d", "dot", "d product", "dp", "sp", "scalar product"):
         return dp(vh)
    elif x in ("cross product", "c", "cross", "c product", "v", "cp", "vp"):
         return cp(vh)

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
                print(f"Invalid input for vector {i + 1}.  Please enter only numbers without spaces.")
        print("So your vectors are:", vectorsholder)
        
        choptype2 = input("Addition, subtraction, magnitudes, multiplication, dot or cross product?_").lower()
        print(optype2(choptype2, vectorsholder))
        
    elif x in ("matrix", "m", "mat"):
        print("Coming soon")
        
    elif x in ("g","guide","gui"):
        print(
            '''
            Vectors:
            Addition: Adds each given vector together (A+B+C...)
            Subtraction: Subtracts each vector from the other sequentially (A-B-C)
            Magnitude: Finds the magnitude of each vector given
            Multiplication (Scalar): Multiplies the scalar onto each vector given (input scalar value later, enter vectors first)
            Dot Product: Scalar products the (first) 2 vectors given
            Cross Product: Vector products the (first) 2 vectors given

            (In construction) Matrices:
            Addition: Adds each given matrix together (A+B+C)
            Subtraction: Subtracts each matrix from the other sequentially (A-B-C)
            Multiplication: Multiplies all given matrices together
            Multiplication (Scalar): Multiplies the scalar onto each matrix given (input scalar value later, enter matrices first)
            Determinant: Finds the determinant of all given matrices
            Transpose: Swaps rows and columns of all given matrices
            Cofactors Matrix: Finds the cofactor matrices of all given matrices
            Adjugate: Finds the transpose of the cofactors matrix
            Inverse: Finds the inverse matrices of all given matrices
            ''')
    
choptype1 = input("Vector or Matrix operation, or Guide (how this calculator works)?_").lower()
optype1(choptype1)

#curr = time.ctime(1627908313.717886)
#print("Current time:", curr)
#just some cool stuff, ignore it
