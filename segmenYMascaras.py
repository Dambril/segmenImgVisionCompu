import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('c.jpg', cv2.IMREAD_COLOR_RGB)
paisaje = cv2.imread('k.jpeg')

#Cositas necesarias para el fondo
#redimensionar para que coincida con la imagen a segmentar, ya que bitwise no acepta diferencias entre tamaños
paisaje = cv2.resize(paisaje, (img.shape[1], img.shape[0]))
#crear lienzo negro del mismo tamaño
alto, ancho, canales = img.shape
imgMask = np.zeros((alto, ancho, canales), np.uint8)

#Preprocesamiento -> 1. Brillo = gamma, 2. Blur = gaussiano
#aplicamos gamma antes de convertir a grises ya que es más efectivo modificar el brillo de la img en RGB
gamma = 0.8 # > 1 aclara, < 1 oscurece, se asignó al evaluar la imagen que seleccionamos al final
invGamma = 1.0 / gamma 
table = np.array([((i/255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
gammaImg = cv2.LUT(img, table) #aplica una Tabla de Consulta a una imagen para transformar rápidamente los valores de los píxeles
gray = cv2.cvtColor(gammaImg, cv2.COLOR_RGB2GRAY) #convertir a grises
blur = cv2.GaussianBlur(gray, (5, 5), 0) #aplicar blur para hacer una mejor segmentación

#Segmentación de la imagen -> Umbralización
#usamos la umbralización automática con otsu en lugar de hacerlo manualmente, la img nos lo permite
_, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) #la persona debe estar blanca, el fondo negro

#Operaciones morfológicas
#op morfológicas para limpiar ruido, aplicamos 2 vistas en clase pero relacionables entre ellas, primero open quita el ruido y con close rellena huecos
kernel = np.ones((5,5), np.uint8)
op = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations=1)
#cerrar los bordes para asegurar que el contorno sea continuo
cl = cv2.morphologyEx(op, cv2.MORPH_CLOSE, kernel, iterations=1)

#Contornos
#se usa retr external para solo detectar a la persona principal
contornos, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#hicimos una copia para no dibujar los contornos en la imagen original
imgContornos = img.copy()
#dibujar contornos
listaAreas = []
for i in range(len(contornos)):
    area = cv2.contourArea(contornos[i])
    listaAreas.append(area)
    cv2.drawContours(imgContornos, contornos, i, (0, 0, 255), 2)

#elegir el contorno más grande, ya que es el que pertenece a la persona
area_max = max(listaAreas)
contorno_max = listaAreas.index(area_max)
cnt_maximo = contornos[contorno_max]

#dibujar un pequeño margen verde para demostrar los contornos detectados
cv2.drawContours(img, contornos, contorno_max, (0, 255, 0), 3)
#dibuja y rellena el contorno principal (objeto de mayor área) sobre la máscara
cv2.drawContours(imgMask, [cnt_maximo], 0, (255, 255, 255), cv2.FILLED)

#aplicar operaciones bitwise para el montaje
#mascara 1: solo el objeto sobre el fondo negro
mascara1 = cv2.bitwise_and(img, imgMask)
#mascara 2: el fondo con un "hueco" donde irá el objeto
imgMaskInv = cv2.bitwise_not(imgMask)
mascara2 = cv2.bitwise_and(paisaje, imgMaskInv)
#sumar ambos para obtener el resultado final
imgFinal = cv2.add(mascara1, mascara2)

#Mostrar resultados
plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(blur, cmap='gray'), plt.title('Preprocesamiento')
plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(cl, cmap='gray'), plt.title('Morfologia')
plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(imgMask, cmap='gray'), plt.title('Mascara gen')
plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(mascara1, cmap='gray'), plt.title('Img segmentada')
plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(imgFinal, cmap='gray'), plt.title('Resultado')
plt.xticks([]), plt.yticks([])

plt.show()
#cv2.imwrite('imgFinal.png', imgFinal)