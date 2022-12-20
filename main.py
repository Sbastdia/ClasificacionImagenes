from codigo.capa_1_convolucion import UnaCapaConvolucion
from codigo.capa_1_convolucion_aumentada import UnaCapaConvolucionAumentada
from codigo.capas_4_convolucion_aumentada import ConvolucionAumentada4

if __name__ == "__main__":
    input("Qué ejemplo quieres ejecutar?")
    opcion=input("1. Una capa de convolución, 2. Una capa de convolución aumentada, 3. 4 capas de convolución aumentada:")

    if opcion == "1":
        UnaCapaConvolucion().ejecutar()
    elif opcion == "2":
        UnaCapaConvolucionAumentada().ejecutar()
    elif opcion == "3":
        ConvolucionAumentada4().ejecutar()
    else:
        print("Opción no válida")

