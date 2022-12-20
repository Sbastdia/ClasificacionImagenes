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




#************************************************************************************
#
# REDES NEURONALES CON 1 CAPA DE CONVOLUCIONES Y UNA CANTIDAD DE IMAGENES AUMENTADA
#
#************************************************************************************

import pandas as pnd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class UnaCapaConvolucionAumentada:


    def __init__(self):
        #Carga de los datos de entrenamiento
        self.observaciones_entrenamiento = pnd.read_csv('codigo/datas/zalando/fashion-mnist_train.csv')
        self.X = np.array(self.observaciones_entrenamiento.iloc[:, 1:])
        #Se crea una tabla de categorías con ayuda del módulo Keras
        self.y = to_categorical(np.array(self.observaciones_entrenamiento.iloc[:, 0]))

    #Definición del largo y ancho de la imagen
    def imagen(self):
        self.LARGO_IMAGEN = 28
        self.ANCHO_IMAGEN = 28

    def distribucionDatos(self):
        #Distribución de los datos de entrenamiento en datos de aprendizaje y datos de validación
        #80 % de datos de aprendizaje y 20 % de datos de validación
        self.X_aprendizaje, self.X_validacion, self.y_aprendizaje, self.y_validacion = train_test_split(self.X, self.y, test_size=0.2, random_state=13)

    def preparacionDatos(self):
        # Se redimensionan las imágenes al formato 28*28 y se realiza una adaptación de escala en los datos de los píxeles
        self.X_aprendizaje = self.X_aprendizaje.reshape(self.X_aprendizaje.shape[0], self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)
        self.X_aprendizaje = self.X_aprendizaje.astype('float32')
        self.X_aprendizaje /= 255

        # Se hace lo mismo con los datos de validación
        self.X_validacion = self.X_validacion.reshape(self.X_validacion.shape[0], self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)
        self.X_validacion = self.X_validacion.astype('float32')
        self.X_validacion /= 255

    def preparacionDatosTest(self):
        #Preparación de los datos de prueba
        self.observaciones_test = pnd.read_csv('codigo/datas/zalando/fashion-mnist_test.csv')

        self.X_test = np.array(self.observaciones_test.iloc[:, 1:])
        self.y_test = to_categorical(np.array(self.observaciones_test.iloc[:, 0]))

        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)
        self.X_test = self.X_test.astype('float32')
        self.X_test /= 255


    def dimensionEntrada(self):
        #Se especifican las dimensiones de la imagen de entrada
        self.dimensionImagen = (self.ANCHO_IMAGEN, self.LARGO_IMAGEN, 1)

    def redNeuronal(self):
        #Se crea la red neuronal capa por capa
        self.redNeurona1Convolucion = Sequential()

        #1- Adición de la capa de convolución que contiene
        #  Capa oculta de 32 neuronas
        #  Un filtro de 3x3 (Kernel) recorriendo la imagen
        #  Una función de activación de tipo ReLU (Rectified Linear Activation)
        #  Una imagen de entrada de 28px * 28 px
        self.redNeurona1Convolucion.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.dimensionImagen))

        #2- Definición de la función de pooling con un filtro de 2px por 2 px
        self.redNeurona1Convolucion.add(MaxPooling2D(pool_size=(2, 2)))

        #3- Adición de una función de ignorancia
        self.redNeurona1Convolucion.add(Dropout(0.2))

        #5 - Se transforma en una sola línea
        self.redNeurona1Convolucion.add(Flatten())

        #6 - Adición de una red neuronal compuesta por 128 neuronas con una función de activación de tipo Relu
        self.redNeurona1Convolucion.add(Dense(128, activation='relu'))

        #7 - Adición de una red neuronal compuesta por 10 neuronas con una función de activación de tipo softmax
        self.redNeurona1Convolucion.add(Dense(10, activation='softmax'))

        #8 - Compilación del modelo

        self.redNeurona1Convolucion.compile(loss=keras.losses.categorical_crossentropy,
                                            optimizer=keras.optimizers.Adam(),
                                            metrics=['accuracy'])


        #9 - Aumento de la cantidad de imágenes

        self.generador_imagenes = ImageDataGenerator(rotation_range=8,
                                    width_shift_range=0.08,
                                    shear_range=0.3,
                                    height_shift_range=0.08,
                                    zoom_range=0.08)


        self.nuevas_imagenes_aprendizaje = self.generador_imagenes.flow(self.X_aprendizaje, self.y_aprendizaje, batch_size=256)
        self.nuevas_imagenes_validacion = self.generador_imagenes.flow(self.X_validacion, self.y_validacion, batch_size=256)

    def aprendizaje(self):
        #10 - Aprendizaje
        self.historico_aprendizaje = self.redNeurona1Convolucion.fit_generator(self.nuevas_imagenes_aprendizaje,
                                                            steps_per_epoch=48000//256,
                                                            epochs=50,
                                                            validation_data=self.nuevas_imagenes_validacion,
                                                            validation_steps=12000//256,
                                                            use_multiprocessing=False,
                                                            verbose=1 )


    def evaluacion(self):
        #11 - Evaluación del modelo
        self.evaluacion = self.redNeurona1Convolucion.evaluate(self.X_test, self.y_test, verbose=0)
        print('Error:', self.evaluacion[0])
        print('Precisión:', self.evaluacion[1])


    #12 - Visualización de la fase de aprendizaje
    def visualizacion(self):

        #Datos de precisión (accuracy)
        plt.plot(self.historico_aprendizaje.history['accuracy'])
        plt.plot(self.historico_aprendizaje.history['val_accuracy'])
        plt.title('Precisión del modelo')
        plt.ylabel('Precisión')
        plt.xlabel('Epoch')
        plt.legend(['Aprendizaje', 'Test'], loc='upper left')
        plt.show()

        #Datos de validación y error
        plt.plot(self.historico_aprendizaje.history['loss'])
        plt.plot(self.historico_aprendizaje.history['val_loss'])
        plt.title('Error')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(['Aprendizaje', 'Test'], loc='upper left')
        plt.show()

    def guardarModelo(self):
        #Guardado del modelo
        # serializar modelo a JSON
        self.modelo_json = self.redNeurona1Convolucion.to_json()
        with open("codigo/modelo/modelo.json", "w") as json_file:
            json_file.write(self.modelo_json)

    def serializarPesos(self):
        # serializar pesos a HDF5
        self.redNeurona1Convolucion.save_weights("codigo/modelo/modelo.h5")
        print("¡Modelo guardado!")



    @staticmethod
    def ejecutar():
        #Se crea un objeto de la clase RedNeuronal
        redNeuronal = UnaCapaConvolucionAumentada()

        redNeuronal.imagen()

        redNeuronal.distribucionDatos()

        redNeuronal.preparacionDatos()

        redNeuronal.preparacionDatosTest()

        #Se llama al método que especifica las dimensiones de la imagen de entrada
        redNeuronal.dimensionEntrada()

        #Se llama al método que crea la red neuronal
        redNeuronal.redNeuronal()

        redNeuronal.aprendizaje()

        #Se llama al método que visualiza la fase de aprendizaje
        redNeuronal.visualizacion()

        #Se llama al método que guarda el modelo
        redNeuronal.guardarModelo()

        #Se llama al método que serializa los pesos
        redNeuronal.serializarPesos()

if __name__ == '__main__':
    UnaCapaConvolucionAumentada.ejecutar()