
# * * * * * * 
# *         * 
# *         * 
# * *     * * 
# * *     * * 
# * * * * * * 

def symmetry(n: int):
   for i in range(n):
        for j in range(n):
            print("*",end='')
        print()
        for j in range(n):
            print(" ",end='')
        print()
        for j in range(n):
            print("*",end="")
        print()
            

symmetry(3)