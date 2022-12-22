import numpy as np 


class Matrix(dict):
    def __init__(self) -> None:
        pass

    def get_matrix_shape(self, name: str) -> tuple:
        """returns the shape of matrix 

        Args
        ----
        name : str
            the name of matrix
        
        """
        
        return self[name].shape


    def isspd(self, name: str) -> bool:
        pass


    def issingular(self, name: str) -> bool:
        pass 





        