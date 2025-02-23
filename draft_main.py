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

def vecSubtract(vh):
    ANS = vh[0][:]
    for vector in vh[1:]:
        for j in range(len(vector)):
            ANS[j] -= vector[j]
    return ANS

def vecAdd(vh):
    vector_length = len(vh[0])
    ANS = [0] * vector_length
    for vector in vh:
        for j in range(vector_length):
            ANS[j] += vector[j]
    return ANS

optype1 = input("Vector or Matrix operation?_").lower()

if optype1 == "vector" or "v" or "vec":

    varnumber = int(input("How many vectors?_")) # how many vectors to do operations on
    vectorholder = [] # holds those vectors in an array
    
    for i in range(varnumber):
        vectorin = input("Enter numbers of your vector (no spaces)_")
        vectorout = [int(char) for char in vectorin]
        vectorholder.append(vectorout) # store the vectors in holder
        print("stored!")
    print("So your vectors are: " + str(vectorholder))


    
    optype2 = input("Addition, subtraction?_").lower()
    if optype2 in ["add", "addition", "a"]:
        print(vecAdd(vectorholder))
    elif optype2 in ["sub", "subtraction", "s"]:
        print(vecSubtract(vectorholder))

