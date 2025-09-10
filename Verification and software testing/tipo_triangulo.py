import pytest

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
            raise TrianguloError("Los lados no forman un triángulo válido (no cumple desigualdad triangular)")
        
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

    def test_escaleno_numeros_grandes(self):
        assert tipo_triangulo(13, 17, 23) == "escaleno"


class TestTrianguloEquilatero:
    def test_equilatero_valido(self):
        assert tipo_triangulo(6, 6, 6) == "equilátero"

    def test_equilatero_numeros_grandes(self):
        assert tipo_triangulo(10, 10, 10) == "equilátero"


class TestTrianguloIsosceles:
    def test_isosceles_valido(self):
        assert tipo_triangulo(5, 5, 8) == "isosceles"

    def test_isosceles_permutacion_primeros_iguales(self):
        assert tipo_triangulo(3, 3, 4) == "isosceles"

    def test_isosceles_permutacion_primero_tercero_iguales(self):
        assert tipo_triangulo(3, 4, 3) == "isosceles"

    def test_isosceles_permutacion_ultimos_iguales(self):
        assert tipo_triangulo(4, 3, 3) == "isosceles"

    def test_isosceles_numeros_grandes(self):
        assert tipo_triangulo(15, 15, 28) == "isosceles"


class TestLadosInvalidos:
    def test_lado_cero(self):
        with pytest.raises(TrianguloError, match="números positivos"):
            tipo_triangulo(0, 5, 5)

    def test_lado_negativo(self):
        with pytest.raises(TrianguloError, match="números positivos"):
            tipo_triangulo(-3, 4, 4)

    def test_varios_lados_negativos(self):
        with pytest.raises(TrianguloError, match="números positivos"):
            tipo_triangulo(-2, -2, 5)

    def test_caso_especial_ceros(self):
        with pytest.raises(TrianguloError, match="números positivos"):
            tipo_triangulo(0, 0, 0)


class TestTriangulosDegenerados:
    def test_suma_dos_lados_igual_tercero_caso1(self):
        with pytest.raises(TrianguloError, match="desigualdad triangular"):
            tipo_triangulo(1, 2, 3)

    def test_suma_dos_lados_igual_tercero_caso2(self):
        with pytest.raises(TrianguloError, match="desigualdad triangular"):
            tipo_triangulo(1, 3, 2)

    def test_suma_dos_lados_igual_tercero_caso3(self):
        with pytest.raises(TrianguloError, match="desigualdad triangular"):
            tipo_triangulo(3, 1, 2)

    def test_suma_dos_lados_igual_tercero_caso4(self):
        with pytest.raises(TrianguloError, match="desigualdad triangular"):
            tipo_triangulo(2, 4, 6)


class TestTriangulosImposibles:
    def test_suma_dos_lados_menor_tercero_caso1(self):
        with pytest.raises(TrianguloError, match="desigualdad triangular"):
            tipo_triangulo(1, 2, 5)

    def test_suma_dos_lados_menor_tercero_caso2(self):
        with pytest.raises(TrianguloError, match="desigualdad triangular"):
            tipo_triangulo(1, 5, 2)

    def test_suma_dos_lados_menor_tercero_caso3(self):
        with pytest.raises(TrianguloError, match="desigualdad triangular"):
            tipo_triangulo(5, 1, 2)

    def test_suma_dos_lados_menor_tercero_caso4(self):
        with pytest.raises(TrianguloError, match="desigualdad triangular"):
            tipo_triangulo(2, 3, 10)


class TestValoresNoEnteros:    
    def test_valores_decimales_escaleno(self):
        assert tipo_triangulo(2.5, 3.7, 4.1) == "escaleno"

    def test_valores_decimales_isosceles(self):
        assert tipo_triangulo(2.5, 2.5, 3.7) == "isosceles"

    def test_valores_decimales_equilatero(self):
        assert tipo_triangulo(3.14, 3.14, 3.14) == "equilátero"

    def test_decimales_invalidos(self):
        with pytest.raises(TrianguloError, match="desigualdad triangular"):
            tipo_triangulo(1.1, 2.2, 5.5)


class TestErrores:
    def test_numero_incorrecto_parametros_pocos(self):
        with pytest.raises(TypeError):
            tipo_triangulo(5, 7)

    def test_numero_incorrecto_parametros_muchos(self):
        with pytest.raises(TypeError):
            tipo_triangulo(5, 7, 8, 9)

    def test_sin_parametros(self):
        with pytest.raises(TypeError):
            tipo_triangulo()


class TestParametrizados:
    @pytest.mark.parametrize("lado1,lado2,lado3,esperado", [
        (5, 7, 9, "escaleno"),
        (6, 6, 6, "equilátero"),
        (5, 5, 8, "isosceles"),
        (3, 3, 4, "isosceles"),
        (3, 4, 3, "isosceles"),
        (4, 3, 3, "isosceles"),
        (2.5, 3.7, 4.1, "escaleno"),
        (2.5, 2.5, 3.7, "isosceles"),
        (3.14, 3.14, 3.14, "equilátero"),
    ])
    def test_triangulos_validos(self, lado1, lado2, lado3, esperado):
        assert tipo_triangulo(lado1, lado2, lado3) == esperado

    @pytest.mark.parametrize("lado1,lado2,lado3,mensaje_error", [
        (0, 5, 5, "números positivos"),
        (-3, 4, 4, "números positivos"),
        (-2, -2, 5, "números positivos"),
        (0, 0, 0, "números positivos"),

        (1, 2, 3, "desigualdad triangular"),
        (1, 3, 2, "desigualdad triangular"),
        (3, 1, 2, "desigualdad triangular"),
        (2, 4, 6, "desigualdad triangular"),
        (1, 2, 5, "desigualdad triangular"),
        (1, 5, 2, "desigualdad triangular"),
        (5, 1, 2, "desigualdad triangular"),
        (2, 3, 10, "desigualdad triangular"),
    ])
    def test_triangulos_invalidos(self, lado1, lado2, lado3, mensaje_error):
        with pytest.raises(TrianguloError, match=mensaje_error):
            tipo_triangulo(lado1, lado2, lado3)


if __name__ == "__main__":
    pytest.main(["-v", __file__])