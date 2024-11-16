# uts_pengolahan_citra_digital
UTS 

Nama : Dimas Alvianto
NIM : 23422009
KELAS : IF22A (PAGI)

import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(image):
   
    height, width, _ = image.shape

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
  
    h = np.zeros(256, dtype=int)
    for i in range(height):
        for j in range(width):
            h[grayscale_image[i, j]] += 1

    
    c = np.cumsum(h)
    c_normalized = (c - c.min()) * 255 / (c.max() - c.min())
    c_normalized = c_normalized.astype('uint8')

   
    equalized_image = np.zeros_like(grayscale_image)
    for i in range(height):
        for j in range(width):
            equalized_image[i, j] = c_normalized[grayscale_image[i, j]]

   
    h2 = np.zeros(256, dtype=int)
    for i in range(height):
        for j in range(width):
            h2[equalized_image[i, j]] += 1

    return grayscale_image, equalized_image, h, h2


image = cv2.imread("walpaper.jpg")
grayscale, equalized, original_hist, equalized_hist = histogram_equalization(image)

plt.figure(figsize=(12, 6))


plt.subplot(2, 2, 1)
plt.title("Original Image (Grayscale)")
plt.imshow(grayscale, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Original Histogram")
plt.bar(range(256), original_hist, color='black')

plt.subplot(2, 2, 3)
plt.title("Equalized Image")
plt.imshow(equalized, cmap='gray')


plt.subplot(2, 2, 4)
plt.title("Equalized Histogram")
plt.bar(range(256), equalized_hist, color='black')

plt.tight_layout()
plt.show()

