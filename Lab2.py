from SimpleCV import Camera, Display, Image, ImageClass ## Cargamos los paquetes de SimpleCV en Python
import cv2 ## Se importan comandos de openCV para cargar imagenes que luego se usaran para determinar el histograma
import matplotlib.pyplot as plt ## Se importan comandos de la libreria matplotlib para graficar el histograma
import numpy as np
img=Image("PapelBlanco.jpg") ##Cargamos archivo de imagen
imggs=img.grayscale() ##Llevamos la imagen a escala de grises
imggs.save("PapelBlanco_gs.jpg") ## Guardamos archivo de imagen en escala de grises
## Repetimos el procedimiento para todas las imagenes tomadas
img2=Image("PapelCuadr.jpg")
img2gs=img2.grayscale()
img2gs.save("PapelCuadr_gs.jpg")
img3=Image("PapelColor.jpg")
img3gs=img3.grayscale()
img3gs.save("PapelColor_gs.jpg")
##A continuacion se cargan las imagenes que se encuentran en escala grises
imgas = cv2.imread('PapelBlanco_gs.jpg',0)
imgas2 = cv2.imread('PapelCuadr_gs.jpg',0)
imgas3 = cv2.imread('PapelColor_gs.jpg',0)


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
plt.title('Histograma Papel de Color anaranjado en escala de grises')
plt.xlabel('Valores de pixeles [0,255]')
plt.ylabel('Numero de pixeles')
plt.xlim([0,255])
plt.grid()
plt.show() ## Muestra los histogramas de las imagenes en escala de grises

##OPCION 1 COLORES
##(red, green, blue)=img1.splitChannels(False)
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
##    plt.title('Histograma de colores Papel Blanco')
##    plt.plot(histof,color)
##    plt.xlim([0,255])
##imagen2=cv2.imread('PapelCuadr.jpg',0)
##color=('b', 'g', 'r')
##plt.figure(2)
##for i, color in enumerate(color):
##    histof2=cv2.calcHist(imagen2,[i], None, [256], [0,256])
##    plt.title('Histograma de colores Papel Cuadriculado')
##    plt.plot(histof2,color)
##    plt.xlim([0,255])
##imagen3=cv2.imread('PapelColor.jpg',0)
##color=('b', 'g', 'r')
##plt.figure(3)
##for i, color in enumerate(color):
##    histof3=cv2.calcHist(imagen3,[i], None, [256], [0,256])
##    plt.title('Histograma de colores Papel de Color')
##    plt.plot(histof3,color)
##    plt.xlim([0,255])
##plt.show()

## Imagenes en escala de grises con comando binarize(), segmentacion manual:
## Segmentacion para imagen con papel blanco

b1=imggs.binarize(150,100,0,5) ## primer segmento desde el valor 0 hasta 150 lo llevamos a un valor 100 de pixel
b12=imggs.binarize(166,150,0,5) ## segundo segmento desde el valor 0 hasta 160 lo llevamos a un valor 150 de pixel
b12r=b12-b1                     ## restamos lo anterior para obtener el segmento numero 2      
b12r.save("Segmentacion1PB.jpg") ## guardamos la imagen del segmento numero 2
## iteramos hasta terminar con la segmentacion para mostrar la imagen resultante 
b13=imggs.binarize(175,166,0,5)
b13r=b13-b12
b13r.save("Segmentacion2PB.jpg")
b14=imggs.binarize(185,175,0,5)
b14r=b14-b13
b14r.save("Segmentacion3PB.jpg")
b15=imggs.binarize(190,185,0,5)
b15r=b15-b14
b15r.save("Segmentacion4PB.jpg")
b16=imggs.binarize(205,190,0,5)
b16r=b16-b15
b16r.save("Segmentacion5PB.jpg")
btotal=b12r+b13r+b14r+b15r+b16r ## reconstruccion de la imagen por segmentacion manual
btotal.save("SegmentacionTOTAL.jpg")
## se repite el procedimiento para la imagen con papel cuadriculado
b2=img2gs.binarize(150,100,0,5)
b22=img2gs.binarize(185,150,0,5)
b22r=b22-b2
b22r.save("Segmentacion1PCr.jpg")
b23=img2gs.binarize(200,185,0,5)
b23r=b23-b22
b23r.save("Segmentacion2PCr.jpg")
b24=img2gs.binarize(215,200,0,5)
b24r=b24-b23
b24r.save("Segmentacion3PCr.jpg")
btotal2=b22r+b23r+b24r
btotal2.save("SegmentacionTOTAL2.jpg")
## se repite el procedimiento para la imagen con papel de color
b3=img3gs.binarize(120,70,0,5)
b32=img3gs.binarize(148,120,0,5)
b32r=b32-b3
b32r.save("Segmentacion1PCol.jpg")
b33=img3gs.binarize(165,148,0,5)
b33r=b33-b32
b33r.save("Segmentacion2PCol.jpg")
b34=img3gs.binarize(177,165,0,5)
b34r=b34-b33
b34r.save("Segmentacion3PCol.jpg")
b35=img3gs.binarize(185,177,0,5)
b35r=b35-b34
b35r.save("Segmentacion4PCol.jpg")
b36=img3gs.binarize(195,185,0,5)
b36r=b36-b35
b36r.save("Segmentacion5PCol.jpg")
b37=img3gs.binarize(205,195,0,5)
b37r=b37-b36
b37r.save("Segmentacion6PCol.jpg")
b38=img3gs.binarize(215,205,0,5)
b38r=b38-b37
b38r.save("Segmentacion7PCol.jpg")
btotal3=b32r+b33r+b34r+b35r+b36r+b37r+b38r
btotal3.save("SegmentacionTOTAL3.jpg")

