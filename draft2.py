import numpy as np
from numpy import linalg as lng
import array
import time

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
            time.sleep(7)
            exit()
    return True

def  addvecs(vh):
    checkvecssize(vh)
    ans = np.array(vh)
    print(np.sum(ans, axis=0))

def subvecs(vh):
    checkvecssize(vh)
    ans = np.array(vh)
    print(reduce(np.subtract, ))

def optype2():
    pass

def optype1(x):
    if x in ("vector", "v", "vec"):
        varnumber = int(input("How many vectors?_"))  # how many vectors to operate on
        vectorsholder = []  # use a list to hold vectors
    elif x in ("matrix", "m", "mat"):
        pass
    elif x in ("g","guide","gui"):
        print(
            '''
            Vectors:
            Addition: Adds each given vector together (A+B+C...)
            Subtraction: Subtracts each vector from the other sequentially (A-B-C)
            Multiplication (Scalar): Multiplies the scalar onto each vector given (input scalar value later, enter vectors first)
            Dot Product: Scalar products the first 2 vectors given
            Cross Product: Vector profucts the first 2 vectors given

            Matrices:
            Addition: Adds each given matrix together (A+B+C)
            Subtraction: Subtracts each matrix from the other sequentially (A-B-C)
            Multiplication: Multiplies all given matrices together
            Multiplication (Scalar): Multiplies the scalar onto each matrix given (input scalar value later, enter matrices first)
            Transpose: Swaps rows and columns of all given matrices
            Determinant: Finds the determinant of all given matrices
            Cofactors Matrix: Finds the cofactor matrices of all given matrices
            Inverse: Finds the inverse matrices of all given matrices
            ''')
    

choptype1 = input("Vector or Matrix operation, or Guide (how this calculator works)?_").lower()
optype1(choptype1)

y=False
while y:
    if optype1 in ("vector", "v", "vec"):
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

        optype2 = input("Addition, subtraction, multiplication (scalar), or cross product?_").lower()
        
        if optype2 in ["add", "addition", "a"]:
            addvecs(vectorsholder)
        elif optype2 in ["sub","s","subtract", "subs"]:
            subvecs(vectorsholder)
        elif optype2 in ["mul","m","multiplication", "mult", "multiply"]:
            pass
        else:
            print("Invalid input. Please enter one of the options.")
        break
        
    elif optype1 in ("matrix", "m", "mat"):
        pass

    elif optype1 in ("g","guide","gui"):
        print(
            '''
            Vectors:
            Addition: Adds each given vector together (A+B+C...)
            Subtraction: Subtracts each vector from the other sequentially (A-B-C)
            Multiplication (Scalar): Multiplies the scalar onto each vector given (input scalar value later, enter vectors first)
            Dot Product: Scalar products the first 2 vectors given
            Cross Product: Vector profucts the first 2 vectors given

            Matrices:
            Addition: Adds each given matrix together (A+B+C)
            Subtraction: Subtracts each matrix from the other sequentially (A-B-C)
            Multiplication: Multiplies all given matrices together
            Multiplication (Scalar): Multiplies the scalar onto each matrix given (input scalar value later, enter matrices first)
            Transpose: Swaps rows and columns of all given matrices
            Determinant: Finds the determinant of all given matrices
            Cofactors Matrix: Finds the cofactor matrices of all given matrices
            Inverse: Finds the inverse matrices of all given matrices
            ''')
        break


#curr = time.ctime(1627908313.717886)
#print("Current time:", curr)
#just some cool stuff, ignore it
