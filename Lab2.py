from SimpleCV import Camera, Display, Image, ImageClass ## Cargamos los paquetes de SimpleCV en Python
import cv2 ## Se importan comandos de openCV para cargar imagenes que luego se usaran para determinar el histograma
import matplotlib.pyplot as plt ## Se importan comandos de la libreria matplotlib para graficar el histograma
import numpy as np
from sklearn.cluster import KMeans ##importamos paquetes de sklearn para usar kmean
import utils ##importamos las funciones creadas a partir de la guia para utilizarlas en el desarrollo del algoritmo kmeans
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
##plt.figure(1)
##plt.hist(imgas.ravel(),256,[0,255]) ## Se asignan valores a los pixeles de la imagen para luego determinar el histograma
##plt.title('Histograma Papel Blanco en escala de grises') 
##plt.xlabel('Valores de pixeles [0,255]')
##plt.ylabel('Numero de pixeles')
##plt.xlim([0,255])
##plt.grid()
##plt.figure(2)
##plt.hist(imgas2.ravel(),256,[0,255])
##plt.title('Histograma Papel Cuadriculado en escala de grises')
##plt.xlabel('Valores de pixeles [0,255]')
##plt.ylabel('Numero de pixeles')
##plt.xlim([0,255])
##plt.grid()
##plt.figure(3)
##plt.hist(imgas3.ravel(),256,[0,255])
##plt.title('Histograma Papel de Color anaranjado en escala de grises')
##plt.xlabel('Valores de pixeles [0,255]')
##plt.ylabel('Numero de pixeles')
##plt.xlim([0,255])
##plt.grid()
##plt.show() ## Muestra los histogramas de las imagenes en escala de grises

##OPCION 1 COLORES
##(red, green, blue)=img.splitChannels(False)
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
b1a=imggs.binarize(150,255,0,5) ##Obtenemos el texto a partir de dejar en blanco todos los pixeles con valor menor igual a 150
b1a.save("Texto_papel_blanco.jpg") ## Guardamos la imagen del texto
b1=imggs.binarize(150,180,0,5) ## primer segmento desde el valor 0 hasta 150 lo llevamos a un valor 100 de pixel
b1ar=-b1 ##aplicamos operacion logica "not" para poder generar la imagen segmentada del mismo tono a la original
b1ar.save("Segmentacion01PB.jpg")
b12=imggs.binarize(166,89,0,5) ## segundo segmento desde el valor 0 hasta 166 lo llevamos a un valor 150 de pixel
b12r=b12-b1 ## restamos el primer segmento al segundo para obtener el segmento numero 2      
b12ra=-b12r ##aplicamos operacion logica "not" para poder generar la imagen segmentada del mismo tono a la original
b12ra.save("Segmentacion1PB.jpg") ## guardamos la imagen del segmento numero 2
## iteramos hasta terminar con la segmentacion para mostrar la imagen resultante 
b13=imggs.binarize(175,80,0,5)
b13r=b13-b12
b13ra=-b13r
b13ra.save("Segmentacion2PB.jpg")
b14=imggs.binarize(185,70,0,5)
b14r=b14-b13
b14ra=-b14r
b14ra.save("Segmentacion3PB.jpg")
b15=imggs.binarize(190,65,0,5)
b15r=b15-b14
b15ra=-b15r
b15ra.save("Segmentacion4PB.jpg")
b16=imggs.binarize(200,55,0,5)
b16r=b16-b15
b16ra=-b16r
b16ra.save("Segmentacion5PB.jpg")
btotal=-(b1+b12r+b13r+b14r+b15r+b16r) ## reconstruccion de la imagen por segmentacion manual
                                      ## se aplica operacion logica "not" para dejar la imagen en el mismo tono a la original
btotal.save("SegmentacionTOTAL.jpg")
## se repite el procedimiento para la imagen con papel cuadriculado
b2=img2gs.binarize(150,180,0,5) ## llevamos los pixeles con valor menor a 150 hasta 75 aprox
b2r=img2gs.binarize(150,255,0,5) ## los llevamos a 1 para posteriormente usarlo con el fin de dejar las cuadriculas 
                                ## tambien utilizado para guardar cuadricula
b2a=-b2
b2a.save("Segmentacion01PCr.jpg")
b2r.save("texto_papel_cuadr.jpg")
b22=img2gs.binarize(190,105,0,5)
b22a=img2gs.binarize(190,255,0,5)
b22r=(b22-b2) ## utilizado para reconstruccion de imagen
b22rr=(b22a-b2) ## utilizado para guardar cuadricula
b22ra=-b22r
b22ra.save("Segmentacion1PCr.jpg")
b23=img2gs.binarize(200,55,0,5)
b23r=(b23-b22) ## utilizado para reconstruccion de imagen
b23ra=-b23r
b23ra.save("Segmentacion2PCr.jpg")
b24=img2gs.binarize(215,40,0,5)
b24r=b24-b23## utilizado para reconstruccion de imagen 
b24ra=-b24r
b24ra.save("Segmentacion3PCr.jpg")
btotal2=-(b2+b22r+b23r+b24r) ## se aplica operacion logica "not" para dejar la imagen en el mismo tono a la original (en escala de grises)
btotal2.save("SegmentacionTOTAL2.jpg")
bCuadr=b22a-b2r
bCuadr.save("SegmentacionCuadr.jpg")
## se repite el procedimiento para la imagen con papel de color
b3a=img3gs.binarize(130,255,0,5)##Obtenemos el texto a partir de dejar en blanco todos los pixeles con valor menor igual a 130
b3a.save("Texto_papel_color.jpg")##Guardamos texto
b3=img3gs.binarize(120,180,0,5) ## utilizado para reconstruccion de imagen
b3r=-b3 
b3r.save("Segmentacion01Pcol.jpg")
b32=img3gs.binarize(148,107,0,5)
b32a=img3gs.binarize(148,255,0,5)
b32r=b32-b3 ## utilizado para reconstruccion de imagen
b32ra=-b32r
b32ra.save("Segmentacion1PCol.jpg")
b33=img3gs.binarize(165,90,0,5)
b33ra=img3gs.binarize(165,255,0,5)
b33r=b33-b32 ## utilizado para reconstruccion de imagen
b33ra=-b33r
b33rr=b33ra-b32a ## utilizado para guardar papel de color
b33ra.save("Segmentacion2PCol.jpg")
b34=img3gs.binarize(177,78,0,5)
b34ra=img3gs.binarize(177,255,0,5)
b34r=b34-b33 ## utilizado para reconstruccion de imagen
b34ra=-b34r
b34rr=b34ra-b33ra ## utilizado para guardar papel de color
b34ra.save("Segmentacion3PCol.jpg")
b35=img3gs.binarize(185,70,0,5)
b35ra=img3gs.binarize(185,255,0,5)
b35r=b35-b34 ## utilizado para reconstruccion de imagen
b35ra=-b35r
b35rr=b35ra-b34ra ## utilizado para guardar papel de color
b35ra.save("Segmentacion4PCol.jpg")
b36=img3gs.binarize(195,60,0,5)
b36r=b36-b35 ## utilizado para reconstruccion de imagen
b36ra=-b36r
b36ra.save("Segmentacion5PCol.jpg")
b37=img3gs.binarize(205,50,0,5)
b37r=b37-b36 ## utilizado para reconstruccion de imagen
b37ra=-b37r
b37ra.save("Segmentacion6PCol.jpg")
b38=img3gs.binarize(215,40,0,5)
b38r=b38-b37 ## utilizado para reconstruccion de imagen
b38ra=-b38r
b38ra.save("Segmentacion7PCol.jpg")
btotal3=-(b3+b32r+b33r+b34r+b35r+b36r+b37r+b38r) ## se reconstruye la imagen en escala de grises mediante segmentacion manual
                ## se aplica operacion logica "not" para dejar la imagen en el mismo tono a la original en escala de grises
btotal3.save("SegmentacionTOTAL3.jpg")
bpapelcol=b33rr+b34rr+b35rr ## se contruye el papel de color en escala de grises
bpapelcol.save("SegmentacionPapCol.jpg")

## Algoritmo de segmentacion kmeans para papel blanco
##image=cv2.imread("PapelBlanco.jpg") ##cargamos imagen original
##image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) ##cambiamos el formato del color de la imagen de BGR a RGB
##image=image.reshape((image.shape[0]*image.shape[1],3))## ajustamos la imagen a una lista de pixeles
##clt=KMeans(n_clusters=2) ##aplicamos Kmeans con 2 clusters para ver blanco y el color del texto
##clt.fit(image)
##hist=utils.centroid_histogram(clt) ## utilizamos funciones creadas mediante codigo de la guia para obtener histograma
##                                   ## y obtener figura de clases entregadas por kmeans
##bar=utils.plot_colors(hist,clt.cluster_centers_)
##plt.figure()
##plt.axis("off")
##plt.imshow(bar)  ## mostramos imagen con las clases de kmeans
##
####repetimos lo anterior para el papel cuadriculado y papel color
####Papel cuadriculado:
##image2=cv2.imread("PapelCuadr.jpg")
##image2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
##image2=image2.reshape((image2.shape[0]*image2.shape[1],3))
##clt2=KMeans(n_clusters=3)
##clt2.fit(image2)
##hist2=utils.centroid_histogram(clt2)
##bar2=utils.plot_colors(hist2,clt2.cluster_centers_)
##plt.figure()
##plt.axis("off")
##plt.imshow(bar2)
####Papel de Color:
##
##image3=cv2.imread("PapelColor.jpg")
##image3=cv2.cvtColor(image3,cv2.COLOR_BGR2RGB)
##image3=image3.reshape((image3.shape[0]*image3.shape[1],3))
##clt3=KMeans(n_clusters=3)
##clt3.fit(image3)
##hist3=utils.centroid_histogram(clt3)
##bar3=utils.plot_colors(hist3,clt3.cluster_centers_)
##plt.figure()
##plt.axis("off")
##plt.imshow(bar3)
##plt.show()

image=cv2.imread("PapelBlanco.jpg")

(h,w)=image.shape[:2]

image=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
image=image.reshape((image.shape[0]*image.shape[1],3))
clt=KMeans(n_clusters=2)
labels=clt.fit_predict(image)
quant=clt.cluster_centers_.astype("uint8")[labels]

quant=quant.reshape((h,w,3))
image=image.reshape((h,w,3))

quant=cv2.cvtColor(quant,cv2.COLOR_LAB2BGR)
image=cv2.cvtColor(image,cv2.COLOR_LAB2BGR)
plt.figure()
plt.imshow(quant)


image2=cv2.imread("PapelCuadr.jpg")

(h2,w2)=image2.shape[:2]

image2=cv2.cvtColor(image2,cv2.COLOR_BGR2LAB)
image2=image2.reshape((image2.shape[0]*image2.shape[1],3))
clt2=KMeans(n_clusters=8)
labels2=clt2.fit_predict(image2)
quant2=clt2.cluster_centers_.astype("uint8")[labels2]

quant2=quant2.reshape((h2,w2,3))
image2=image2.reshape((h2,w2,3))

quant2=cv2.cvtColor(quant2,cv2.COLOR_LAB2BGR)
image2=cv2.cvtColor(image2,cv2.COLOR_LAB2BGR)
plt.figure()
plt.imshow(quant2)

image3=cv2.imread("PapelColor.jpg")

(h3,w3)=image3.shape[:2]

image3=cv2.cvtColor(image3,cv2.COLOR_BGR2LAB)
image3=image3.reshape((image3.shape[0]*image3.shape[1],3))
clt3=KMeans(n_clusters=3)
labels3=clt3.fit_predict(image3)
quant3=clt3.cluster_centers_.astype("uint8")[labels3]

quant3=quant3.reshape((h3,w3,3))
image3=image3.reshape((h3,w3,3))

quant3=cv2.cvtColor(quant3,cv2.COLOR_LAB2BGR)
image3=cv2.cvtColor(image3,cv2.COLOR_RGB2BGR)


plt.figure()
plt.imshow(quant3)
##2
image4=cv2.imread("PapelColor.jpg")

(h4,w4)=image4.shape[:2]

image4=cv2.cvtColor(image4,cv2.COLOR_BGR2LAB)
image4=image4.reshape((image4.shape[0]*image4.shape[1],3))
clt4=KMeans(n_clusters=2)
labels4=clt4.fit_predict(image4)
quant4=clt4.cluster_centers_.astype("uint8")[labels4]


quant4=quant4.reshape((h4,w4,3))


quant4=cv2.cvtColor(quant4,cv2.COLOR_LAB2BGR)

plt.figure()
plt.imshow(quant4)


final=(quant4-quant3)

plt.figure()
plt.imshow(final)
plt.show()
