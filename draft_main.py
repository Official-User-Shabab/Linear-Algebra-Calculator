# A vector is an array
# A matrix is also an array, but of more arrays (a collection of vectors)

# For example, if there are 3 vectors, A, B, C, then matrix1 is a 3x3 matrix consisting of those vectors

vectorA = [1, 2, 3]
vectorB = [3, 2, 1]
vectorC = [1, 0, 1]

matrix1 = [vectorA, vectorB, vectorC]   # Issue: It is better to handle each column as a vector, so how to imagine this in the form of an array?

#        COLUMNS
#  R      0 1 2 
#  O   0 [1 3 1]  This is matrix1
#  W   1 |2 2 0|  When referencing to a matrix, it must be as (column, row)
#  s   2 [3 1 1]  Example: To get element 0: matrix1[2, 1]
