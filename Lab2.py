#! /usr/bin/python
from SimpleCV import Camera, Display, Image, DrawingLayer, ImageClass ## Cargamos los paquetes de SimpleCV en Python
import numpy as np
import cv2
import matplotlib.pyplot as plt
img=Image("PapelBlanco.jpg") ##Cargamos archivo de imagen
imggs=img.grayscale() ##Llevamos la imagen a escala de grises
imggs.save("PapelBlanco(gs).jpg") ## Guardamos archivo de imagen en escala de grises
## Repetimos lo anterior para todas las imagenes tomadas
img2=Image("PapelCuadriculado.jpg")
img2gs=img2.grayscale()
img2gs.save("PapelCuadriculado(gs).jpg")
img3=Image("PapelCuadriculadocolor.jpg") 
img3gs=img3.grayscale()
img3gs.save("PapelCuadriculadocolor(gs).jpg")
img4=Image("PapelColor.jpg")
img4gs=img4.grayscale()
img4gs.save("PapelColor(gs).jpg")
hist1=imggs.histogram()
hist4=img4gs.histogram()
##imagen1=cv2.imread('/home/pi/Lab2/PapelBlanco(gs).jpg')
##color=('b', 'g', 'r')
##for i, color in enumerate(color):
##    histof=cv2.calcHist(imagen1,[i], None, [256], [0,256])
##    plt.title('Histograma de colores')
##    plt.plot(histof,color)
##    plt.xlim([50,150])
##plt.show()
imgas = cv2.imread('PapelBlanco(gs).jpg')
imgas2 = cv2.imread('PapelCuadriculado(gs).jpg')
imgas3 = cv2.imread('PapelCuadriculadocolor(gs).jpg')
imgas4 = cv2.imread('PapelColor(gs).jpg')
plt.figure(1)
plt.hist(imgas.ravel(),256,[0,255],normed=True)
plt.title('Histograma Normalizado Papel Blanco en escala de grises')
plt.xlabel('Valores de pixeles [0,255]')
plt.grid()
plt.figure(2)
plt.hist(imgas2.ravel(),256,[0,255],normed=True)
plt.title('Histograma Normalizado Papel Cuadriculado en escala de grises')
plt.xlabel('Valores de pixeles [0,255]')
plt.grid()
plt.figure(3)
plt.hist(imgas3.ravel(),256,[0,255],normed=True)
plt.title('Histograma Normalizado Papel Cuadriculado color en escala de grises')
plt.xlabel('Valores de pixeles [0,255]')
plt.grid()
plt.figure(4)
plt.hist(imgas4.ravel(),256,[0,255],normed=True)
plt.title('Histograma Normalizado Papel de Color verde en escala de grises')
plt.xlabel('Valores de pixeles [0,255]')
plt.grid()
plt.show()

