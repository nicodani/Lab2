#! /usr/bin/python
from SimpleCV import Camera, Display, Image ## Cargamos los paquetes de SimpleCV en Python
import cv2 ## Se importan comandos de openCV para cargar imagenes que luego se usaran para determinar el histograma
import matplotlib.pyplot as plt ## Se importan comandos de la libreria matplotlib para graficar el histograma
img=Image("PapelBlanco.jpg") ##Cargamos archivo de imagen
imggs=img.grayscale() ##Llevamos la imagen a escala de grises
imggs.save("PapelBlanco(gs).jpg") ## Guardamos archivo de imagen en escala de grises
## Repetimos el procedimiento para todas las imagenes tomadas
img2=Image("PapelCuadriculado.jpg")
img2gs=img2.grayscale()
img2gs.save("PapelCuadriculado(gs).jpg")
img3=Image("PapelCuadriculadocolor.jpg") 
img3gs=img3.grayscale()
img3gs.save("PapelCuadriculadocolor(gs).jpg")
img4=Image("PapelColor.jpg")
img4gs=img4.grayscale()
img4gs.save("PapelColor(gs).jpg")
##A continuacion se cargan las imagenes que se encuentran en escala grises
imgas = cv2.imread('PapelBlanco(gs).jpg',0)
imgas2 = cv2.imread('PapelCuadriculado(gs).jpg',0)
imgas3 = cv2.imread('PapelCuadriculadocolor(gs).jpg',0)
imgas4 = cv2.imread('PapelColor(gs).jpg',0)
## Se grafican los histogramas de las imagenes respectivas
plt.figure(1)
plt.hist(imgas.ravel(),256,[0,255]) ## Se asignan valores a los pixeles de la imagen para luego determinar el histograma
plt.title('Histograma Papel Blanco en escala de grises') 
plt.xlabel('Valores de pixeles [0,255]')
plt.ylabel('Numero de pixeles')
plt.xlim([0,255])
plt.grid()
plt.figure(2)
plt.hist(imgas2.ravel(),256,[0,255])
plt.title('Histograma Papel Cuadriculado en escala de grises')
plt.xlabel('Valores de pixeles [0,255]')
plt.ylabel('Numero de pixeles')
plt.xlim([0,255])
plt.grid()
plt.figure(3)
plt.hist(imgas3.ravel(),256,[0,255])
plt.title('Histograma Papel Cuadriculado color en escala de grises')
plt.xlabel('Valores de pixeles [0,255]')
plt.ylabel('Numero de pixeles')
plt.xlim([0,255])
plt.grid()
plt.figure(4)
plt.hist(imgas4.ravel(),256,[0,255])
plt.title('Histograma Papel de Color verde en escala de grises')
plt.xlabel('Valores de pixeles [0,255]')
plt.ylabel('Numero de pixeles')
plt.xlim([0,255])
plt.grid()
plt.show() ## Muestra los histogramas de las imagenes en escala de grises

##OPCION 1 COLORES
##(red, green, blue)=img4.splitChannels(False)
##redhist=red.histogram(255)
##greenhist=green.histogram(255)
##bluehist=blue.histogram(255)
##plt.figure(1)
##plt.plot(redhist)
##plt.figure(2)
##plt.plot(greenhist)
##plt.figure(3)
##plt.plot(bluehist)
##plt.show()

##OPCION 2 COLORES
##imagen1=cv2.imread('PapelBlanco.jpg',0)
##color=('b', 'g', 'r')
##plt.figure(1)
##for i, color in enumerate(color):
##    histof=cv2.calcHist(imagen1,[i], None, [256], [0,256])
##    plt.title('Histograma de colores')
##    plt.plot(histof,color)
##    plt.xlim([0,255])
##imagen2=cv2.imread('PapelCuadriculado.jpg',0)
##color=('b', 'g', 'r')
##plt.figure(2)
##for i, color in enumerate(color):
##    histof2=cv2.calcHist(imagen2,[i], None, [256], [0,256])
##    plt.title('Histograma de colores')
##    plt.plot(histof2,color)
##    plt.xlim([0,255])
##imagen3=cv2.imread('PapelColor.jpg',0)
##color=('b', 'g', 'r')
##plt.figure(3)
##for i, color in enumerate(color):
##    histof3=cv2.calcHist(imagen3,[i], None, [256], [0,256])
##    plt.title('Histograma de colores')
##    plt.plot(histof3,color)
##    plt.xlim([0,255])
##plt.show()

## Imagenes en blanco y negro con comando binarize()
b1=imggs.binarize(thresh=-1, maxv=255, blocksize=0, p=5)
b1.show()
b2=img2gs.binarize(thresh=-1, maxv=255, blocksize=0, p=5)
b2.show()
b3=img3gs.binarize(thresh=-1, maxv=255, blocksize=0, p=5)
b3.show()

