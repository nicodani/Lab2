#! /usr/bin/python
from SimpleCV import Camera, Display, Image ## Cargamos los paquetes de SimpleCV en Python
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
