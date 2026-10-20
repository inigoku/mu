import decimal

def calculate_pi(precision):
    """
    Calcula el número pi con la precisión especificada utilizando
    una versión simplificada del algoritmo de Chudnovsky.
    """
    # Establecemos la precisión del contexto de decimal
    # Sumamos unos pocos dígitos extra para evitar errores de redondeo al final
    decimal.getcontext().prec = precision + 3

    C = 426880 * decimal.Decimal(10005).sqrt()
    K = 6
    M = 1
    X = 1
    L = 13591409
    S = L

    # Para 100 decimales, unas 15-20 iteraciones son más que suficientes
    # ya que cada término de la serie añade aproximadamente 14 dígitos precisos
    for i in range(1, 15):
        M = (M * (K**3 - 16*K)) // (i**3)
        L += 545140134
        X *= -262537412640768000
        S += decimal.Decimal(M * L) / X
        K += 12

    pi = C / S

    # Retornamos el resultado truncado a la precisión exacta solicitada
    return str(pi)[:precision + 2] # +2 para incluir el "3."

if __name__ == "__main__":
    digits = 100
    result = calculate_pi(digits)
    print(f"Los primeros {digits} decimales de Pi son:\n\n{result}")
