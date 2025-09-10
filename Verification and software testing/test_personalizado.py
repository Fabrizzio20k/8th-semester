import pytest
import sys


tipos = ["isosceles", "equilátero", "escaleno"]


class TrianguloError(Exception):
    pass


def es_triangulo_valido(a, b, c):
    if a <= 0 or b <= 0 or c <= 0:
        return False
    return (a + b > c and a + c > b and b + c > a)


def tipo_triangulo(lado1, lado2, lado3):
    if not es_triangulo_valido(lado1, lado2, lado3):
        if lado1 <= 0 or lado2 <= 0 or lado3 <= 0:
            raise TrianguloError("Los lados deben ser números positivos")
        else:
            raise TrianguloError(
                "Los lados no forman un triángulo válido (no cumple desigualdad triangular)")

    if lado1 == lado2 == lado3:
        return tipos[1]
    elif lado1 == lado2 or lado2 == lado3 or lado1 == lado3:
        return tipos[0]
    else:
        return tipos[2]


class TestTrianguloEscaleno:
    def test_escaleno_valido(self):
        assert tipo_triangulo(5, 7, 9) == "escaleno"

    def test_escaleno_triangulo_rectangulo(self):
        assert tipo_triangulo(3, 4, 5) == "escaleno"


class TestBigValuesWithSysMaxsize:
    def test_escaleno_numeros_grandes(self):
        assert tipo_triangulo(sys.maxsize, sys.maxsize - 1,
                              sys.maxsize - 2) == "escaleno"


class TestWithErrors:
    def test_numero_incorrecto_parametros_pocos(self):
        with pytest.raises(TrianguloError):
            tipo_triangulo(5, 7, sys.maxsize + 1)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
