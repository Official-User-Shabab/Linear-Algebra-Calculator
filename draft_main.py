Avector = []   # A vector is an array

# A matrix is also an array, but of more arrays (a collection of vectors)
Amatrix = [
  [],
  [],
  [] ]

# For example, if there are 3 vectors, A, B, C, then matrix1 is a 3x3 matrix consisting of those vectors

vectorA = [1, 2, 3]
vectorB = [3, 2, 1]
vectorC = [1, 0, 1]
matrix1 = [vectorA, vectorB, vectorC]   # Issue: It is better to handle each column as a vector, so how to imagine this in the form of an array?
