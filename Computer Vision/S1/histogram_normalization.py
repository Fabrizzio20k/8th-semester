import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import requests

url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4gMjOzuXBi93S43pS2IlcaAgexW_fSOEj-A&s"

response = requests.get(url)
image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
color_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
original_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

fig, ((ax_img1, ax_hist1), (ax_img2, ax_hist2)
      ) = plt.subplots(2, 2, figsize=(15, 10))
plt.subplots_adjust(bottom=0.15)


def histogram_equalization(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    cdf = np.cumsum(hist)

    cdf_norm = cdf / (image.shape[0] * image.shape[1])

    new_intensities = np.round(cdf_norm * 255).astype(np.uint8)

    equalized_image = new_intensities[image]

    return equalized_image, hist, cdf_norm


def normalize_histogram(hist):
    return hist / hist.sum()


def update_contrast(val):
    alpha = slider.val

    adjusted_image = cv2.convertScaleAbs(original_image, alpha=alpha, beta=0)

    ax_img1.clear()
    ax_img1.imshow(adjusted_image, cmap='gray')
    ax_img1.set_title(f'Imagen Original (Contraste: {alpha:.2f})')
    ax_img1.axis('off')

    ax_hist1.clear()
    hist_orig = cv2.calcHist([adjusted_image], [0], None, [256], [0, 256])
    hist_norm_orig = normalize_histogram(hist_orig)
    ax_hist1.plot(hist_norm_orig, color='blue', linewidth=2)
    ax_hist1.set_xlim([0, 256])
    ax_hist1.set_xlabel('Intensidad')
    ax_hist1.set_ylabel('Frecuencia Normalizada')
    ax_hist1.set_title('Histograma Original')
    ax_hist1.grid(True, alpha=0.3)

    equalized_image, _, _ = histogram_equalization(adjusted_image)

    ax_img2.clear()
    ax_img2.imshow(equalized_image, cmap='gray')
    ax_img2.set_title('Imagen Ecualizada')
    ax_img2.axis('off')

    ax_hist2.clear()
    hist_eq = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
    hist_norm_eq = normalize_histogram(hist_eq)
    ax_hist2.plot(hist_norm_eq, color='red', linewidth=2)
    ax_hist2.set_xlim([0, 256])
    ax_hist2.set_xlabel('Intensidad')
    ax_hist2.set_ylabel('Frecuencia Normalizada')
    ax_hist2.set_title('Histograma Ecualizado')
    ax_hist2.grid(True, alpha=0.3)

    fig.canvas.draw()


ax_img1.imshow(original_image, cmap='gray')
ax_img1.set_title('Imagen Original')
ax_img1.axis('off')

hist_orig = cv2.calcHist([original_image], [0], None, [256], [0, 256])
hist_norm_orig = normalize_histogram(hist_orig)
ax_hist1.plot(hist_norm_orig, color='blue', linewidth=2)
ax_hist1.set_xlim([0, 256])
ax_hist1.set_xlabel('Intensidad')
ax_hist1.set_ylabel('Frecuencia Normalizada')
ax_hist1.set_title('Histograma Original')
ax_hist1.grid(True, alpha=0.3)

equalized_image, _, _ = histogram_equalization(original_image)

ax_img2.imshow(equalized_image, cmap='gray')
ax_img2.set_title('Imagen Ecualizada')
ax_img2.axis('off')

hist_eq = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
hist_norm_eq = normalize_histogram(hist_eq)
ax_hist2.plot(hist_norm_eq, color='red', linewidth=2)
ax_hist2.set_xlim([0, 256])
ax_hist2.set_xlabel('Intensidad')
ax_hist2.set_ylabel('Frecuencia Normalizada')
ax_hist2.set_title('Histograma Ecualizado')
ax_hist2.grid(True, alpha=0.3)

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Contraste', 0.5, 3.0, valinit=1.0, valstep=0.1)
slider.on_changed(update_contrast)

plt.show()
