#! /usr/bin/python
from SimpleCV import Image, Display, Camera ## Cargamos paquetes de simpleCV en python
import time ## Se importa tiempo a python para hacer uso del comando sleep()
c=Camera() ## Se le asigna una variable a la camara
time.sleep(2) ## Se hace esperar dos segudnos a la camara, para evitar alto brillo en la captura
img=c.getImage() ## Se captura la imagen
img.save("PapelBlanco.jpg") ## Se guarda la imagen en un archivo
