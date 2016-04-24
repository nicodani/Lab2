from SimpleCV import Image, Display, Camera ## Cargamos paquetes de simpleCV en python
from sklearn.cluster import KMeans ##Cargamos KMeans
from sklearn.cluster import MiniBatchKMeans ##Cargamos Kmeans que es más rapido pero menos preciso
import cv2 ##Cargamos para trabajar en kmeans con opencv
import numpy as np ## Cargamos numpy para trabajar sobre imagenes
import matplotlib.pyplot as plt ##Cargamos para poder plotear
import time ## Se importa tiempo a python para hacer uso del comando sleep()
c=Camera() ## Se le asigna una variable a la camara
time.sleep(2) ## Se hace esperar dos segundos a la camara, para evitar alto brillo en la captura
img=c.getImage() ## Se captura la imagen
img.save("imagen.jpg") ##guardamos imagen para cargarla con comando cv2.imread() posteriormente
fig=img.show() 
time.sleep(5) ##Mostramos imagen por 5 segundo y cerramos
fig.quit()
print "Seleccione una de las siguientes opciones: "
print "\t1.- Papel Blanco"
print "\t2.- Papel Cuadriculado" ##Creamos menu para realizar la eleccion del papel sobre el que se trabaja
print "\t3.- Papel de Color"

while True:

    opc=raw_input("Ingrese el numero de una opcion: ")
    if opc=="1":
        ## Segmentacion Manual
        imggs=img.grayscale() ##llevamos a escala de grises la imagen
        b1=imggs.binarize(150,255,0,5) ##Obtenemos el texto
        fig1=b1.show()
        time.sleep(5) ##Mostramos imagen por 5 segundo y cerramos
        fig1.quit()
        ## Segmentacion mediante Kmeans
        image=cv2.imread("imagen.jpg") ##cargamos imagen
        (h,w)=image.shape[:2] ##guardamos el largo y ancho de imagen
        image=image.reshape((image.shape[0]*image.shape[1],3)) ## arreglamos imagen para trabajar
        clt=MiniBatchKMeans(n_clusters=2) ## aplicamos Kmeans
        labels=clt.fit_predict(image) ## Predecimos los valores de pixeles
        quant=clt.cluster_centers_.astype("uint8")[labels] ##obtenemos resultado de kmeans
        quant=quant.reshape((h,w,3)) ## arreglamos imagen para mostrar
        plt.figure()
        plt.axis("off")
        plt.imshow(quant) ## mostramos imagen con algoritmo kmeans
        plt.show()
        break
    elif opc=="2":
        ##segmentacion Manual
        imggs=img.grayscale()
        b1=imggs.binarize(150,255,0,5) ##Obtenemos el texto
        b1a=imggs.binarize(190,255,0,5) ##Segundo segmento
        b1r=b1a-b1 ##obtenemos segmento que posee la cuadricula
        fig1=b1.show()
        time.sleep(5)##Mostramos imagen por 5 segundo y cerramos
        fig1.quit()
        fig2=b1r.show()
        time.sleep(5)##Mostramos imagen por 5 segundo y cerramos
        fig2.quit()
        ##Segmentacion Kmeans (como se hizo en el primer caso)
        ##Texto
        image=cv2.imread("imagen.jpg")
        (h,w)=image.shape[:2]
        image=image.reshape((image.shape[0]*image.shape[1],3))
        clt=KMeans(n_clusters=2)
        labels=clt.fit_predict(image)
        quant=clt.cluster_centers_.astype("uint8")[labels]
        quant=quant.reshape((h,w,3))
        plt.figure()
        plt.axis("off")
        plt.imshow(quant)
        ##Cuadriculado, utilizamos ambos
        image=cv2.imread("imagen.jpg")
        (h,w)=image.shape[:2]
        image=image.reshape((image.shape[0]*image.shape[1],3))
        clt=KMeans(n_clusters=7)
        labels=clt.fit_predict(image)
        quant2=clt.cluster_centers_.astype("uint8")[labels]
        quant2=quant2.reshape((h,w,3))
        fin=quant2-quant ## restamos las dos imagenes resultantes para obtener cuadriculado
        plt.figure()
        plt.axis("off")
        plt.imshow(fin)
        plt.show()
        break
    elif opc=="3":
        ##segmentacion Manual
        imggs=img.grayscale()
        b1=imggs.binarize(130,255,0,5) ##Obtenemos el texto
        b1a=imggs.binarize(165,255,0,5) ## primer segmento papel color
        b1aa=imggs.binarize(185,255,0,5) ## Segundo segmento papel color
        b1r=b1aa-b1a ##obtenemos segmento que posee el papel de color
        fig1=b1.show()
        time.sleep(5)##Mostramos imagen por 5 segundo y cerramos
        fig1.quit()
        fig2=b1r.show()
        time.sleep(5)##Mostramos imagen por 5 segundo y cerramos
        fig2.quit()
        ## Algoritmo Kmeans
        ## papel color
        image=cv2.imread("PapelColor.jpg")
        (h,w)=image.shape[:2]
        image=image.reshape((image.shape[0]*image.shape[1],3))
        clt=KMeans(n_clusters=2)
        labels=clt.fit_predict(image)
        quant=clt.cluster_centers_.astype("uint8")[labels]
        quant=quant.reshape((h,w,3))
        plt.figure()
        plt.axis("off")
        plt.imshow(quant)
        ##texto
        image=cv2.imread("PapelColor.jpg")
        (h,w)=image.shape[:2]
        image=image.reshape((image.shape[0]*image.shape[1],3))
        clt=KMeans(n_clusters=3)
        labels=clt.fit_predict(image)
        quant2=clt.cluster_centers_.astype("uint8")[labels]
        quant2=quant2.reshape((h,w,3))
        fin=quant2-quant ## restamos imagenes para obtener el texto del papel
        plt.figure()
        plt.axis("off")
        plt.imshow(fin)
        plt.show()
        break
    else:
        print "opcion no valida"

