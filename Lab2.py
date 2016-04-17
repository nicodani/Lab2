#! /usr/bin/python
from SimpleCV import Camera, Display, Image
import time
c=Camera()
img=c.getImage()
img.save("PapelBlanco.jpg")

