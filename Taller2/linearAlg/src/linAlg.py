import math
from typing import List, Union, Tuple, Optional
import numpy as np


class Vector:
    """
    Clase para representar vector2 manipular vectores.
    
    Un vector es una lista de números que puede representar
    puntos en el espacio, direcciones, o cualquier secuencia ordenada de valores.
    """
    
    def __init__(self, components: List[Union[int, float]]):
        """
        Inicializa un vector con sus componentes.
        
        Args:
            components: Lista de números que representan las componentes del vector
        """

        self.components = components
    
    def __str__(self) -> str:
        """Representación en string del vector."""

        return str(self.components)
    
    def __repr__(self) -> str:
        """Representación detallada del vector."""
        return self.components
    
    def __len__(self) -> int:
        """Retorna la dimensión del vector."""
        return len(self.components)
    
    def __getitem__(self, index: int) -> Union[int, float]:
        """Permite acceder a los componentes del vector usando índices."""
        return self.components[index]
    
    def __setitem__(self, index: int, value: Union[int, float]):
        """Permite modificar componentes del vector usando índices."""
        self.components[index] = value 
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """Suma de vectores usando el operador +."""
        if len(self.components) == len(other): 
            vector1 = self.components
            vector2 = other.components

            result =  [ int(a + b) for a,b in zip(vector1, vector2)]
            return result

        else :
             print('mismatching between lenghts of vectors. please use in the input vectors of the same lenght')       
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """Resta de vectores usando el operador -."""
        if len(self.components) == len(other): 
            vector1 = self.components
            vector2 = other.components

            result = [int(a - b) for a,b in zip(vector1, vector2)]
            return result

        else :
             print('mismatching between lenghts of vectors. please use in the input vectors of the same lenght')     
    
    def __mul__(self, scalar: Union[int, float]) -> 'Vector':
        """Multiplicación por escalar usando el operador *."""
        
        vector1 = self.components
        scalar = scalar

        result = [ int(i * scalar)  for i  in vector1] 
        return result       

    def __rmul__(self, scalar: Union[int, float]) -> 'Vector':
        """Multiplicación por escalar (orden invertido)."""
      
        vector1 = self.components
        scalar = scalar

        result = [i * scalar for i  in vector1] 
        return result.reverse()

        
    
    def __truediv__(self, scalar: Union[int, float]) -> 'Vector':
        """División por escalar usando el operador /."""

        vector1 = self.components
        scalar = int(scalar)
        new_scalar = 1/scalar

        result = [ int(i * new_scalar)  for i  in vector1] 
        return result.reverse()

    
    
    def __eq__(self, other: 'Vector') -> bool:
        """Igualdad entre vectores usando el operador ==."""
        if not isinstance(other, Vector):
           return False
        return self.components == other.components
    
    def __ne__(self, other: 'Vector') -> bool:
        """Desigualdad entre vectores usando el operador !=."""
        if not isinstance(other, Vector):
           return False
        return self.components != other.components
    
    @property

    def magnitude(self) -> float:
        """Calcula vector2 retorna la magnitud (norma) del vector."""
        vector = self.components 
        norma = [i **2 for i in vector]
        result = sum(norma)**0.5
        return result
    
    @property

    def unit_vector(self) -> 'Vector':
        """Retorna el vector unitario (normalizado)."""
        mag = self.magnitude
        vector = self.components

        if mag == 0:
               raise ValueError("can't calculate a unit vector from a null vector")
        normalized = [i / mag for i in vector]
        return Vector(normalized)
    
    def dot(self, other: 'Vector') -> float:
        """
        Calcula el producto punto con otro vector.
        
        Args:
            other: Otro vector para el producto punto
            
        Returns:
            El producto punto como un número
        """
        vector1 = self.components
        vector2 = other.components

        if len(vector1) != len(vector2):
            raise ValueError("vectors must have the same dimension")

        operation = [a*b for a,b in zip (vector1, vector2)]
        result = sum(operation)

        return result
    
    def cross(self, other: 'Vector') -> 'Vector':
        """
        Calcula el producto cruz con otro vector (solo para vectores 3D).
        
        Args:
            other: Otro vector para el producto cruz
            
        Returns:
            Un nuevo vector resultado del producto cruz
        """
        vector1 = self.components
        vector2 = other.components

        if len(vector1) != 3 or len(vector2) != 3:
                   raise ValueError("Cross product it's only calculated for vectors in R^3")
        else: 
            return [vector1[1]*vector2[2] - vector1[2]*vector2[1],
                    vector1[2]*vector2[0] - vector1[0]*vector2[2],
                    vector1[0]*vector2[1] - vector1[1]*vector2[0]]
       
    
    def angle_with(self, other: 'Vector') -> float:
        """
        Calcula el ángulo entre este vector vector2 otro.
        
        Args:
            other: Otro vector
            
        Returns:
            El ángulo en radianes
        """
        
        
        
        dot_product = self.dot(other)
        mag_product = self.magnitude * other.magnitude
 

        if mag_product == 0:
            raise ValueError("Can't calculate an angle of a null vector")
        
        cos_theta = max(-1, min(1, dot_product / mag_product))
        return math.acos(cos_theta)
    
class Matrix:
    """
    Clase para representar vector2 manipular matrices.
    
    Una matriz es una colección rectangular de números organizados en filas vector2 columnas.
    """
    
    def __init__(self, data: List[List[Union[int, float]]]):
        """
        Inicializa una matriz con sus datos.
        
        Args:
            data: Lista de listas que representa las filas de la matriz
        """
        self.data = [list(row) for row in data] 
        self.rows = len(data)
        self.cols = len(data[0]) if isinstance(data[0], list) else len(data)

    
    def __str__(self) -> str:
        """Representación en string de la matriz."""
        return str(self.data)
    
    def __repr__(self) -> str:
        """Representación detallada de la matriz."""
        return self.data
    
    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> Union[List[Union[int, float]], Union[int, float]]:
        """Permite acceder a filas o elementos específicos de la matriz."""
        if isinstance(key, int):
                return self.data[key]
        elif isinstance(key, tuple) and len(key) == 2:
            i, j = key
            return self.data[i][j]
        else:
            raise TypeError("index must be an int or tuple of two ints")
    
    def __setitem__(self, key: Union[int, Tuple[int, int]], value: Union[List[Union[int, float]], Union[int, float]]):
        """Permite modificar filas o elementos específicos de la matriz."""
        if isinstance(key, int):
                self.data[key] = value
        elif isinstance(key, tuple) and len(key) == 2:
                i, j = key
                self.data[i][j] = value
        else:
            raise TypeError("index must be an int or tuple of two ints")
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Suma de matrices usando el operador +."""
        if not isinstance(other, Matrix):
            raise TypeError("just can be operated using another instance of matrix")
        
        if len(self.data) != len(other.data) or any(len(r1) != len(r2) for r1, r2 in zip(self.data, other.data)):
            raise ValueError("matrix must have the same dimetions to be operated")
        
        result = [
            [a + b for a, b in zip(row1, row2)]
            for row1, row2 in zip(self.data, other.data)
        ]
        
        return Matrix(result)
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Resta de matrices usando el operador -."""
        if not isinstance(other,Matrix):
            raise TypeError("just can be operated using another instance of matrix")
        
       
        if len(self.data) != len(other.data) or any(len(r1) != len(r2) for r1, r2 in zip(self.data, other.data)):
            raise ValueError("matrix must have the same dimetions to be operated")
        
        result = [
            [a - b for a, b in zip(row1, row2)]
            for row1, row2 in zip(self.data, other.data)
        ]
        
        return Matrix(result)
    
    def __mul__(self, other: Union['Matrix', 'Vector', int, float]) -> Union['Matrix', 'Vector']:
        """Multiplicación de matrices/vectores/escalares usando el operador *."""
        # Escalar
        if isinstance(other, (int, float)):
            return Matrix([[elem * other for elem in row] for row in self.data])
        
        # Otra matriz
        if isinstance(other,Matrix):
            if self.cols != other.rows:
                raise ValueError(" incompatible dimensions for matrix multiplication")
            result = [[sum(a*b for a, b in zip(self_row, col)) 
                    for col in zip(*other.data)] 
                    for self_row in self.data]
            return Matrix(result)

        # Vector
        if isinstance(other, Vector):
            if self.cols != len(other.components):
                raise ValueError("incompatible dimensions for matrix multiplication with a vector")
            result = [sum(a*b for a, b in zip(self_row, other.components)) 
                    for self_row in self.data]
            return Vector(result)

        raise TypeError(f"Can't multiplicate")
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Matrix':
        """Multiplicación por escalar (orden invertido)."""
        return self * scalar
    
    def __eq__(self, other: 'Matrix') -> bool:
        """Igualdad entre matrices usando el operador ==."""
        return isinstance(other, Matrix) and self.data == other.data
    
    def __ne__(self, other: 'Matrix') -> bool:
        """Desigualdad entre matrices usando el operador !=."""
        return not self.__eq__(other)
    
    @property
    def num_rows(self) -> int:
        """Retorna el número de filas de la matriz."""
        return len(self.data)
    
    @property
    def num_columns(self) -> int:
        """Retorna el número de columnas de la matriz."""
        return self.cols
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Retorna las dimensiones de la matriz como (filas, columnas)."""
        return (self.rows, self.cols)
    
    @property
    def T(self) -> 'Matrix':
        """Retorna la transpuesta de la matriz."""
        transpuesta = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(transpuesta)
    
    @property
    def trace(self) -> Union[int, float]:
        """Calcula vector2 retorna la traza de la matriz (suma de elementos diagonales)."""
        if not self.is_square():
           raise ValueError("Trace it's only for squared matrix")
        return sum(self.data[i][i] for i in range(self.rows))
    
    @property
    def determinant(self) -> Union[int, float]:
        """Calcula vector2 retorna el determinante de la matriz."""
        n = self.rows
        if n == 1:
            return self.data[0][0]
        elif n == 2:
            return self.data[0][0]*self.data[1][1] - self.data[0][1]*self.data[1][0]
        elif n == 3:
            a,b,c = self.data[0]
            d,e,f = self.data[1]
            g,h,i = self.data[2]
            return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
        else:
            raise NotImplementedError("Determinant it's only available for 3x3 matrix")
    
    @property
    def inverse(self) -> 'Matrix':
        """Calcula vector2 retorna la matriz inversa."""
        if not self.is_square():
            raise ValueError("Inverse it's only defined for squared matrix")

        n = self.rows
        det = self.determinant
        if det == 0:
            raise ValueError("it has no inverse")

        if n == 1:
            return Matrix([[1/self.data[0][0]]])

        if n == 2:
            a,b = self.data[0]
            c,d = self.data[1]
            return Matrix([[ d/det, -b/det],
                           [-c/det,  a/det]])

        if n == 3:
            m = self.data
            cof = [[ (m[(i+1)%3][(j+1)%3]*m[(i+2)%3][(j+2)%3] - 
                      m[(i+1)%3][(j+2)%3]*m[(i+2)%3][(j+1)%3]) for j in range(3)] for i in range(3)]
            adj = list(map(list, zip(*cof)))  # transpuesta de cofactores
            inv = [[adj[i][j]/det for j in range(3)] for i in range(3)]
            return Matrix(inv)

        raise NotImplementedError("inverse it's only available for 3x3 matrix")
    
    def is_square(self) -> bool:
        """Verifica si la matriz es cuadrada."""
        return self.rows == self.cols
    
    def is_symmetric(self) -> bool:
        """Verifica si la matriz es simétrica."""
        if not self.is_square():
            return False
        return self.data == self.T.data
    
    def is_diagonal(self) -> bool:
        """Verifica si la matriz es diagonal."""
        if not self.is_square():
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if i != j and self.data[i][j] != 0:
                    return False
        return True
    
    def get_row(self, index: int) -> 'Vector':
        """
        Obtiene una fila específica como vector.
        
        Args:
            indevector1: Índice de la fila
            
        Returns:
            Vector con los elementos de la fila
        """
        return Vector(self.data[index])
    
    def get_column(self, index: int) -> 'Vector':
        """
        Obtiene una columna específica como vector.
        
        Args:
            indevector1: Índice de la columna
            
        Returns:
            Vector con los elementos de la columna
        """
        return Vector([self.data[i][index] for i in range(self.rows)])


# =============================================================================
# FUNCIONES DE VECTOR
# =============================================================================

def dot_product(v1: Vector, v2: Vector) -> float:
    """
    Calcula el producto punto entre dos vectores.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        El producto punto como un número
    """

    if len(v1) != len(v2):
        raise ValueError("vectors must have the same dimension")

    operation = [a*b for a,b in zip (v1, v2)]
    result = sum(operation)

    return result


def magnitude(v: Vector) -> float:
    """
    Calcula la magnitud (norma) de un vector.
    
    Args:
        v: El vector
        
    Returns:
        La magnitud del vector
    """
    norma = [i **2 for i in v]
    result = sum(norma)**0.5
    return result

def normalize(v: Vector) -> Vector:
    """
    Normaliza un vector (lo convierte en vector unitario).
    
    Args:
        v: El vector a normalizar
        
    Returns:
        Un nuevo vector normalizado
    """
    mag = magnitude(v)
   
    if mag == 0:
            raise ValueError("can't calculate a unit vector from a null vector")
    normalized = [i / mag for i in v]
    return normalized


def cross_product(v1: Vector, v2: Vector) -> Vector:
    """
    Calcula el producto cruz entre dos vectores 3D.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        Un nuevo vector resultado del producto cruz
    """
    vector1 = v1
    vector2 = v2

    if len(vector1) != 3 or len(vector2) != 3:
                   raise ValueError("Cross product it's only calculated for vectors in R^3")
    else: 
            return [vector1[1]*vector2[2] - vector1[2]*vector2[1],
                    vector1[2]*vector2[0] - vector1[0]*vector2[2],
                    vector1[0]*vector2[1] - vector1[1]*vector2[0]]
       


def angle_between(v1: Vector, v2: Vector) -> float:
    """
    Calcula el ángulo entre dos vectores.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        El ángulo en radianes
    """

    vector1 = v1
    vector2 = v2
        
          
    dot = dot_product(vector1, vector2)
    mag_product = magnitude(vector1)* magnitude(vector2)


    if mag_product == 0:
        raise ValueError("Can't calculate an angle of a null vector")
    
    cos_theta = max(-1, min(1, dot / mag_product))
    return math.acos(cos_theta)


# =============================================================================
# FUNCIONES DE MATRIX
# =============================================================================

def scale(matrix: Matrix, scalar: Union[int, float]) -> Matrix:
    """Multiplica una matriz por un escalar."""
    result = [[scalar * val for val in row] for row in matrix.data]
    return Matrix(result)


def add(m1: Matrix, m2: Matrix) -> Matrix:
    """Suma dos matrices del mismo tamaño."""
    if m1.shape != m2.shape:
        raise ValueError("Matrices must have the same dimensions for addition")
    result = [
        [m1[i][j] + m2[i][j] for j in range(m1.num_columns)]
        for i in range(m1.num_rows)
    ]
    return Matrix(result)


def subtract(m1: Matrix, m2: Matrix) -> Matrix:
    """Resta dos matrices del mismo tamaño."""
    if m1.shape != m2.shape:
        raise ValueError("Matrices must have the same dimensions for addition")
    result = [
        [m1[i][j] - m2[i][j] for j in range(m1.num_columns)]
        for i in range(m1.num_rows)
    ]
    return Matrix(result)


def vector_multiply(matrix: Matrix, vector: Vector) -> Vector:
    """Multiplica una matriz por un vector."""
   # Verificamos dimensiones
    if matrix.cols != len(vector.components):
        raise ValueError("Matrix columns must equal vector length")

    # Multiplicación matriz × vector
    result = [
        sum(matrix.data[i][j] * vector.components[j] for j in range(matrix.cols))
        for i in range(matrix.rows)
    ]
    return Vector(result)


def matrix_multiply(m1: Matrix, m2: Matrix) -> Matrix:
    """Multiplica dos matrices."""
    if m1.cols != m2.rows:
        raise ValueError("Matrix dimensions are incompatible for multiplication")

    result = [
        [
            sum(m1.data[i][k] * m2.data[k][j] for k in range(m1.cols))
            for j in range(m2.cols)
        ]
        for i in range(m1.rows)
    ]
    return Matrix(result)

def transpose(matrix: Matrix) -> Matrix:
    """Calcula la transpuesta de una matriz."""
    result = [list(row) for row in zip(*matrix.data)]
    return Matrix(result)


def determinant(matrix: Matrix) -> Union[int, float]:
    """Calcula el determinante de una matriz 2x2."""
    if matrix.rows != matrix.cols:
        raise ValueError("Determinant is only defined for square matrices")
    if matrix.rows != 2:
        raise NotImplementedError("Determinant only implemented for 2x2 matrices")
    return matrix.data[0][0] * matrix.data[1][1] - matrix.data[0][1] * matrix.data[1][0]


def inverse(matrix: Matrix) -> Matrix:
    """Calcula la inversa de una matriz 2x2."""
    if matrix.rows != 2 or matrix.cols != 2:
        raise NotImplementedError("Inverse only implemented for 2x2 matrices")

    det = determinant(matrix)
    if det == 0:
        raise ValueError("It has no inverse")

    result = [
        [ matrix.data[1][1] / det, -matrix.data[0][1] / det],
        [-matrix.data[1][0] / det,  matrix.data[0][0] / det],
    ]
    return Matrix(result)


def identity_matrix(size: int) -> Matrix:
    """Crea una matriz identidad."""
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    return Matrix(result)


def zeros_matrix(rows: int, columns: int) -> Matrix:
    """Crea una matriz de ceros."""
    result = [[0 for _ in range(columns)] for _ in range(rows)]
    return Matrix(result)


def ones_matrix(rows: int, columns: int) -> Matrix:
    """Crea una matriz de unos."""
    result = [[1 for _ in range(columns)] for _ in range(rows)]
    return Matrix(result)