from codigo.capa_1_convolucion import UnaCapaConvolucion
from codigo.capa_1_convolucion_aumentada import UnaCapaConvolucionAumentada
from codigo.capas_4_convolucion_aumentada import ConvolucionAumentada4
from codigo.clasificacion import ClasificacionImagenes

if __name__ == "__main__":
    print("Qué ejemplo quieres ejecutar?")
    opcion=input("1. Una capa de convolución, 2. Una capa de convolución aumentada,\n3. Cuatro capas de convolución aumentada, 4. Clasificacion final del modelo:")

    if opcion == "1":
        UnaCapaConvolucion().ejecutar()
    elif opcion == "2":
        UnaCapaConvolucionAumentada().ejecutar()
    elif opcion == "3":
        ConvolucionAumentada4().ejecutar()
    elif opcion == "4":
        ClasificacionImagenes().ejecutar()
    else:
        print("Opción no válida")

