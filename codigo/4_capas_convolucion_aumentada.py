#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 12 - Clasificación de imágenes
#
# Módulos necesarios:
#   PANDAS 0.24.2
#   KERAS 2.2.4
#   PILOW 6.0.0
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


#************************************************************************************
#
# RED NEURONAL DE 4 CAPAS DE CONVOLUCIONES CON UNA CANTIDAD DE IMAGENES EN AUMENTO
#
#************************************************************************************

import pandas as pnd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.preprocessing.image import ImageDataGenerator
import time
import matplotlib.pyplot as plt
class ConvolucionAumentada4:
    def __init__(self):
#Definición del largo y ancho de la imagen
        self.LARGO_IMAGEN = 28
        self.ANCHO_IMAGEN = 28

#Carga de los datos de entrenamiento
        self.observaciones_entrenamiento = pnd.read_csv('código cap12/datas/zalando/fashion-mnist_train.csv')

#Solo se guardan las características "píxeles"
    def pixel(self):
        self.X = np.array(self.observaciones_entrenamiento.iloc[:, 1:])

        #Se crea una tabla de categorías con la ayuda del módulo Keras
    def tabla(self):
        self.y = to_categorical(np.array(self.observaciones_entrenamiento.iloc[:, 0]))

        #Distribución de los datos de entrenamiento en datos de aprendizaje y datos de validación
        #80 % de datos de aprendizaje y 20 % de datos de validación
    def datosAprendizaje(self):
        self.X_aprendizaje, self.X_validacion, self.y_aprendizaje, self.y_validacion = train_test_split(self.X, self.y, test_size=0.2, random_state=13)

    def redimensionApr(self):
        #Se redimensionan las imágenes al formato 28*28 y se realiza una adaptación de escala en los datos de los píxeles
        self.X_aprendizaje = self.X_aprendizaje.reshape(self.X_aprendizaje.shape[0], self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)
        self.X_aprendizaje = self.X_aprendizaje.astype('float32')
        self.X_aprendizaje /= 255

    def redimensionVal(self):
        #Se hace lo mismo con los datos de validación
        self.X_validacion = self.X_validacion.reshape(self.X_validacion.shape[0], self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)
        self.X_validacion = self.X_validacion.astype('float32')
        self.X_validacion /= 255
#Hasta aquí el entrenamiento, ahora se hace la prueba
    def datosPrueba(self):
        #Preparación de los datos de prueba
        self.observaciones_test = pnd.read_csv('código cap12/datas/zalando/fashion-mnist_test.csv')

        self.X_test = np.array(self.observaciones_test.iloc[:, 1:])
        self.y_test = to_categorical(np.array(self.observaciones_test.iloc[:, 0]))

        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)
        self.X_test = self.X_test.astype('float32')
        self.X_test /= 255
        #Se especifican las dimensiones de la imagen de entrada
        self.dimensionImagen = (self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)

#Se crea la red neuronal capa por capa
    def redNeuronal(self):
        self.redNeuronas4Convolucion = Sequential()
        self.redNeuronas4Convolucion.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.dimensionImagen))
        self.redNeuronas4Convolucion.add(BatchNormalization())

        self.redNeuronas4Convolucion.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.redNeuronas4Convolucion.add(BatchNormalization())
        self.redNeuronas4Convolucion.add(MaxPooling2D(pool_size=(2, 2)))
        self.redNeuronas4Convolucion.add(Dropout(0.25))

        self.redNeuronas4Convolucion.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.redNeuronas4Convolucion.add(BatchNormalization())
        self.redNeuronas4Convolucion.add(MaxPooling2D(pool_size=(2, 2)))
        self.redNeuronas4Convolucion.add(Dropout(0.25))

        self.redNeuronas4Convolucion.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.redNeuronas4Convolucion.add(BatchNormalization())
        self.redNeuronas4Convolucion.add(MaxPooling2D(pool_size=(2, 2)))
        self.redNeuronas4Convolucion.add(Dropout(0.25))

        self.redNeuronas4Convolucion.add(Flatten())
        self.redNeuronas4Convolucion.add(Dense(512, activation='relu'))
        self.redNeuronas4Convolucion.add(BatchNormalization())
        self.redNeuronas4Convolucion.add(Dropout(0.5))
        self.redNeuronas4Convolucion.add(Dense(10, activation='softmax'))

    def Modelo(self):
        #8 - Compilación del modelo
        self.redNeuronas4Convolucion.compile(loss=keras.losses.categorical_crossentropy,
                                    optimizer=keras.optimizers.Adam(),
                                    metrics=['accuracy'])

    def Imagenes(self):
        #9 - Aumento de la cantidad de imágenes
        self.generador_imagenes = ImageDataGenerator(rotation_range=8,
                            width_shift_range=0.08,
                            shear_range=0.3,
                            height_shift_range=0.08,
                            zoom_range=0.08)


        self.nuevas_imagenes_aprendizaje = self.generador_imagenes.flow(self.X_aprendizaje, self.y_aprendizaje, batch_size=256)
        self.nuevas_imagenes_validacion = self.generador_imagenes.flow(self.X_validacion, self.y_validacion, batch_size=256)

    def Aprendizaje(self):
        #10 - Aprendizaje
        start = time.clock();
        self.historico_aprendizaje = self.redNeuronas4Convolucion.fit_generator(self.nuevas_imagenes_aprendizaje,
                                                    steps_per_epoch=48000//256,
                                                    epochs=50,
                                                    validation_data=self.nuevas_imagenes_validacion,
                                                    validation_steps=12000//256,
                                                    use_multiprocessing=False,
                                                    verbose=1 )
        stop = time.clock();
        print("Tiempo de aprendizaje = "+str(stop-start))

    def Evaluacion(self):
        #11 - Evaluación del modelo
        evaluacion = self.redNeuronas4Convolucion.evaluate(self.X_test, self.y_test, verbose=0)
        print('Error:', evaluacion[0])
        print('Precisión:', evaluacion[1])


#12 - Visualización de la fase de aprendizaje
    def Precision(self):
        plt.plot(self.historico_aprendizaje.history['accuracy'])
        plt.plot(self.historico_aprendizaje.history['val_accuracy'])
        plt.title('Precisión del modelo')
        plt.ylabel('Precisión')
        plt.xlabel('Epoch')
        plt.legend(['Aprendizaje', 'Test'], loc='upper left')
        plt.show()
    def ValidacionyError(self):
        plt.plot(self.historico_aprendizaje.history['loss'])
        plt.plot(self.historico_aprendizaje.history['val_loss'])
        plt.title('Error')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(['Aprendizaje', 'Test'], loc='upper left')
        plt.show()

    def SaveModel(self):
        #Guardado del modelo
        # serializar modelo a JSON
        modelo_json = self.redNeuronas4Convolucion.to_json()
        with open("modelo/modelo_4convoluciones.json", "w") as json_file:
            json_file.write(modelo_json)

        # serializar pesos a HDF5
        self.redNeuronas4Convolucion.save_weights("modelo/modelo_4convoluciones.h5")
        print("¡Modelo guardado!")
    def Train(self):
        self.pixel()
        self.tabla()
        self.datosAprendizaje()
        self.redimensionApr()
        self.redimensionVal()
    def Test(self):
        self.datosPrueba()
        self.redNeuronal()
        self.Modelo()
        self.Imagenes()
        self.Aprendizaje()
        self.Evaluacion()
    def Visualizacion(self):
        self.Precision()
        self.ValidacionyError()
    @staticmethod
    def Ejecutar():
        modelo = ConvolucionAumentada4()
        modelo.Train()
        modelo.Test()
        modelo.Visualizacion()
        modelo.SaveModel()

if __name__ == '__main__':
    ConvolucionAumentada4.Ejecutar()