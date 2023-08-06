import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import csv

# open an image file
image_file = 'desired_image.jpg'
image = Image.open(image_file)

# convert to grey values
grey_image = image.convert('L')
grey_values = np.array(grey_image)

# apply fourier transformation to grayscale image
fourier_transform = np.fft.fft2(grey_values)
shifted_fourier = np.fft.fftshift(fourier_transform)
grey_spectrum = np.abs(shifted_fourier)

# export grey values to a CSV file
csv_file_grey = 'grey_values.csv'
with open(csv_file_grey, 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(grey_values)

# import the same CSV file 
imported_data = np.genfromtxt(csv_file_grey, delimiter = ',')

# find the diagonal values of the imported array and apply fourier transformation
main_diagonal = np.diagonal(imported_data)
fourier_transform_diagonal = np.fft.fft(main_diagonal)
diagonal_spectrum = np.abs(fourier_transform_diagonal)

# safe the diagonal values
csv_file_diagonal = 'grey_values_diagonal.csv'
with open(csv_file_diagonal, 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(main_diagonal)

# plot results
plt.figure(figsize=(9, 6))
gs = plt.GridSpec(3, 3)

# the original image
plt.subplot(gs[0, 0])
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# the grayscale image
plt.subplot(gs[0, 1])
plt.imshow(grey_image, cmap = 'gray')
plt.title("Grayscale Image")
plt.axis("off")

# the fourier transformation of the grayscale image
plt.subplot(gs[0, 2])
plt.imshow(np.log(1 + grey_spectrum), cmap = 'gray')
plt.title("FT of Grayscale")
plt.axis("off")

# the original diagonal values
plt.subplot(gs[1, :])
plt.title('Original Main Diagonal')
plt.plot(main_diagonal)

# the fourier transformation of the original diagonal values
plt.subplot(gs[2, :])
plt.title('FT of Main Diagonal')
plt.plot(diagonal_spectrum)

plot_graph = 'Or_diagonal_and_FT.png'
plt.savefig(plot_graph)

plt.tight_layout()
plt.show()
