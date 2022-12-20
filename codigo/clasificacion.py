#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 12 - Clasificación de imágenes
#
# Módulos necesarios:
#   PANDAS 0.24.2
#   KERAS 2.2.4
#   PILLOW 6.0.0
#   SCIKIT-LEARN 0.20.3
#   NUMPY 1.16.3
#   MATPLOTLIB : 3.0.3
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------


#----------------------------
# CARGA DEL MODELO
#----------------------------

from keras.models import model_from_json
from PIL import Image, ImageFilter
import numpy as np
class ClasificacionImagenes:
    def __init__(self):
        #Carga de la descripción del modelo
        archivo_json = open('codigo/modelo/modelo_4convoluciones.json', 'r')
        self.modelo_json = archivo_json.read()
        archivo_json.close()
        #Definición de las categorías de clasificación
        self.clases = ["Una camiseta/top","Un pantalón","Un jersey","Un vestido","Un abrigo","Una sandalia","Una camisa","Zapatillas","Un bolso","Botines"]


    def PesosModelo(self):
        #Carga de la descripción de los pesos del modelo
        self.modelo = model_from_json(self.modelo_json)
        # Cargar pesos en el modelo nuevo
        self.modelo.load_weights("codigo/modelo/modelo_4convoluciones.h5")



#---------------------------------------------
# CARGA Y TRANSFORMACIÓN DE UNA IMAGEN
#---------------------------------------------

    def Imagen(self):
        #Carga de la imagen
        self.imagen = Image.open("codigo/imagenes/zapatilla.jpg").convert('L')

        #Dimensión de la imagen
        self.largo = float(self.imagen.size[0])
        self.alto = float(self.imagen.size[1])

    def ImagenNueva(self):
        #Creación de una imagen nueva
        self.nuevaImagen = Image.new('L', (28, 28), (255))

    def RedimensionarImagen(self):
        #Redimensionamiento de la imagen
        #La imagen es más larga que alta, la ponemos a 20 píxeles
        if self.largo > self.alto:
            #Se calcula la relación de ampliación entre la altura y el largo
            relacionAltura = int(round((20.0 / self.largo * self.alto), 0))
            if (relacionAltura == 0):
                self.nAltura = 1

            #Redimensionamiento
            img = self.imagen.resize((20, relacionAltura), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            #Posición horizontal
            posicion_alto = int(round(((28 - relacionAltura) / 2), 0))

            self.nuevaImagen.paste(img, (4, posicion_alto))  # pegar imagen redimensionada en lienzo en blanco
        else:

            relacionAltura = int(round((20.0 / self.alto * self.largo), 0))  # redimensionar anchura según relación altura
            if (relacionAltura == 0):  # caso raro pero el mínimo es 1 píxel
                relacionAltura = 1

            #Redimensionamiento
            img = self.imagen.resize((relacionAltura, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

            #Cálculo de la posición vertical
            altura_izquierda = int(round(((28 - relacionAltura) / 2), 0))
            self.nuevaImagen.paste(img, (altura_izquierda, 4))

    def Pixeles(self):
        #Recuperación de los píxeles
        self.pixeles = list(self.nuevaImagen.getdata())

    def Tabla(self):
        #Normalización de los píxeles
        tabla = [(255 - x) * 1.0 / 255.0 for x in self.pixeles]

        #Transformación de la tabla en tabla numpy
        img = np.array(tabla)

        #Se transforma la tabla lineal en imagen 28x20
        self.imagen_test = img.reshape(1, 28, 28, 1)

    def Prediccion(self):
        #Predicción de la imagen
        #prediccion = modelo.predict_classes(imagen_test)
        #esto ha quedado obsoleto en la version 2.6 de tensorflow
        prediccion = self.modelo(self.imagen_test)
        prediccion_clases = np.argmax(prediccion, axis=-1)
        print()
        print("La imagen es: "+self.clases[prediccion_clases[0]])
        print()

    def Probabilidades(self):
        #Extracción de las probabilidades
        probabilidades = self.modelo.predict(self.imagen_test)

        i = 0
        for clase in self.clases:
            print(clase + ": " + str((probabilidades[0][i] * 100)) + "%")
            i = i + 1

    @staticmethod
    def ejecutar():
        clasificacionImagenes = ClasificacionImagenes()
        clasificacionImagenes.PesosModelo()
        clasificacionImagenes.Imagen()
        clasificacionImagenes.ImagenNueva()
        clasificacionImagenes.RedimensionarImagen()
        clasificacionImagenes.Pixeles()
        clasificacionImagenes.Tabla()
        clasificacionImagenes.Prediccion()
        clasificacionImagenes.Probabilidades()

if __name__ == '__main__':
    ClasificacionImagenes.ejecutar()