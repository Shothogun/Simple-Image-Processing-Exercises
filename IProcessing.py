import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def convert_to_gray(image):

	width, height, dimensions = image.shape
	gray_image = np.zeros((height, width), np.uint8)

	b, g, r = cv2.split(image)

	w = np.array([[[0.3, 0.55, 0.15]]])
	gray_image = cv2.convertScaleAbs(np.sum(image * w, axis=2))

	return gray_image


def blur_box_convolve(image, size=3):
	pad = (size - 1) // 2

	# Blur box kernel
	kernel = np.ones((size, size), np.float32) / size ** 2

	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	image = np.uint8(signal.convolve2d(np.float32(image), kernel))

	return image


def blur_gaussian_convolve(image, size=3):
	pad = (size - 1) // 2

	# Gaussian kernel
	if size == 3:
			kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16

	elif size == 5:
			kernel = (
					np.array(
							[
									[1, 4, 7, 4, 1],
									[4, 16, 26, 16, 4],
									[7, 26, 41, 26, 7],
									[4, 16, 26, 16, 4],
									[1, 4, 7, 4, 1],
							],
							np.float32,
					)
					/ 273
			)

	else:
			kernel = (
					np.array(
							[
									[0, 0, 1, 2, 1, 0, 0],
									[0, 3, 13, 22, 13, 3, 0],
									[1, 13, 59, 97, 59, 13, 1],
									[2, 22, 97, 159, 97, 22, 2],
									[1, 13, 59, 97, 59, 13, 1],
									[0, 3, 13, 22, 13, 3, 0],
									[0, 0, 1, 2, 1, 0, 0],
							],
							np.float32,
					)
					/ 1003
			)

	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	image = np.uint8(signal.convolve2d(image, kernel))

	return image


def blur_motion_convolve(image):
	pad = 3

	# Blur motion box kernel
	kernel = (
			np.array(
					[
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
					],
					np.float32,
			)
			/ 9
	)

	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	image = np.uint8(signal.convolve2d(image, kernel))

	return image


def high_pass_convolve(image):
	pad = 1
	kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32) / 9

	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
	image = np.uint8(signal.convolve2d(image, kernel)) + 128

	return image


def fft_filtering(image, kernel, size=3):
	pad = (size - 1) // 2
	Hi, Wi = image.shape

	# Image Border
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

	Hk, Wk = kernel.shape

	# Pad Kernel
	sz = (image.shape[0] - kernel.shape[0], image.shape[1] - kernel.shape[1])
	kernel = np.pad(
			kernel,
			(((sz[0] + 1) // 2, sz[0] // 2), ((sz[1] + 1) // 2, sz[1] // 2)),
			"constant",
	)

	# Shifts kernel origin to 1st image
	# pixel(top-left corner)
	kernel = np.fft.ifftshift(kernel)

	f_kernel = np.fft.fft2(kernel)
	f_kernel = np.fft.fftshift(f_kernel)

	# Image Spectrum
	image_spectrum = np.fft.fft2(image)
	image_spectrum = np.fft.fftshift(image_spectrum)

	# Blur Computation
	image_spectrum *= f_kernel

	# Converts Spectrum back to image
	image_spectrum = np.fft.ifftshift(image_spectrum)
	img_back = np.fft.ifft2(image_spectrum)
	img_back = np.real(img_back)

	return img_back


def blur_box_FFT(image, size=3):
	# Blur Kernel
	kernel = np.ones((size, size), np.float32) / 9

	img_back = fft_filtering(image, kernel, size)

	return img_back


def blur_gaussian_FFT(image, size=3):
	# Blur Kernel
	if size == 3:
			kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16

	elif size == 5:
			kernel = (
					np.array(
							[
									[1, 4, 7, 4, 1],
									[4, 16, 26, 16, 4],
									[7, 26, 41, 26, 7],
									[4, 16, 26, 16, 4],
									[1, 4, 7, 4, 1],
							],
							np.float32,
					)
					/ 273
			)

	else:
			kernel = (
					np.array(
							[
									[0, 0, 1, 2, 1, 0, 0],
									[0, 3, 13, 22, 13, 3, 0],
									[1, 13, 59, 97, 59, 13, 1],
									[2, 22, 97, 159, 97, 22, 2],
									[1, 13, 59, 97, 59, 13, 1],
									[0, 3, 13, 22, 13, 3, 0],
									[0, 0, 1, 2, 1, 0, 0],
							],
							np.float32,
					)
					/ 1003
			)

	img_back = fft_filtering(image, kernel, size)

	return img_back


def blur_motion_FFT(image, size=3):
	# Blur Kernel
	kernel = (
			np.array(
					[
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
							[0, 0, 0, 1, 0, 0, 0],
					],
					np.float32,
			)
			/ 9
	)

	img_back = fft_filtering(image, kernel, 7)

	return img_back


def ideal_filter_FFT(image, size=5):
	center_x, center_y = image.shape
	kernel = np.ones((size, size), np.float32)

	img_back = fft_filtering(image, kernel, size)

	return img_back

def image_enhancement(image, enhance):
	size = 3
	pad = 1

	Hi, Wi = image.shape

	image = ideal_filter_FFT(image)

	residue_image = np.ones((Hi+4*pad, Wi+4*pad), np.uint8) - image
	image = image + residue_image*enhance

	return image

def add_gaussian_noise(image, mean = 0, var=100):
	row, col= image.shape
	sigma = var ** 0.5

	gaussian = np.random.normal(mean, sigma, (row, col))
	noisy = image + gaussian
	cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1) 
	noisy = noisy.astype(np.uint8)

	return noisy

def add_salt_pepper_noise(image, amount = 0.004, s_vs_p = 0.1 ):
	noisy = np.copy(image)
	
	num_salt = np.ceil(amount * image.size * s_vs_p)
	num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
	
 	#Add Salt noise
	coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy.shape]
	noisy[coords[0], coords[1]] = 1

	# Add Pepper noise
	coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy.shape]
	noisy[coords[0], coords[1]] = 0

	return noisy

if __name__ == "__main__":

	# Read Image
	original_image = cv2.imread("me.JPG", -1)
	original_image = cv2.resize(original_image, (300, 400))

	# Convert do Gray
	gray_image = convert_to_gray(original_image)

	# Gets images's spectrum
	image_spectrum = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)

	image_spectrum = np.fft.fftshift(image_spectrum)

	original_magnitude_spectrum = 20 * np.log(
			cv2.magnitude(image_spectrum[:, :, 0], image_spectrum[:, :, 1])
	)

	# 1. Image convolve Blurring
	box_blur_image = blur_box_convolve(gray_image, 5)
	gaussian_blur_image = blur_gaussian_convolve(gray_image, 7)
	motion_blur_image = blur_motion_convolve(gray_image)

	# 2. Image FFT Blurring
	fft_box_blur_image = blur_box_FFT(gray_image, 5)
	fft_gaussian_blur_image = blur_gaussian_FFT(gray_image, 7)
	fft_motion_blur_image = blur_motion_FFT(gray_image)

	# 3. Ideal filter
	fft_ideal_filter = ideal_filter_FFT(gray_image, 10)

	# 4. High pass filter
	high_pass_filtered_image = high_pass_convolve(gray_image)

	# 5. Image Low Pass enhancement
	enhanced_image = image_enhancement(gray_image, 4)

	# 6. Filter Gaussian noise with low pass
	denoised_gaussian_image = add_gaussian_noise(gray_image,0, 50)
	filtered_gaussian_image = ideal_filter_FFT(gray_image, 2)

	# 7. Filter Salt&Pepper noise
	denoised_salt_pepper_image = add_salt_pepper_noise(gray_image,0.1)
	filtered_salt_pepper_image = cv2.medianBlur(denoised_salt_pepper_image, 3)

	# 1. Convolve filtering
	plt.figure(1)
	plt.suptitle("Convolve filtering")
	plt.subplot(141), plt.imshow(gray_image, cmap="gray")
	plt.title("Original image"), plt.xticks([]), plt.yticks([])
	plt.subplot(142), plt.imshow(box_blur_image, cmap="gray")
	plt.title("Box Blur Image"), plt.xticks([]), plt.yticks([])
	plt.subplot(143), plt.imshow(gaussian_blur_image, cmap="gray")
	plt.title("Gaussian Blur Image"), plt.xticks([]), plt.yticks([])
	plt.subplot(144), plt.imshow(motion_blur_image, cmap="gray")
	plt.title("Motion Blur Image"), plt.xticks([]), plt.yticks([])

	# 2. Frequency spectrum
	plt.figure(2)
	plt.suptitle("Frequency domain")
	plt.subplot(121), plt.imshow(gray_image, cmap="gray")
	plt.title("Input Image"), plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(original_magnitude_spectrum, cmap="gray")
	plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])

	# 3. FFT filtering plot
	plt.figure(3)
	plt.suptitle("FFT filtering domain")
	plt.subplot(141), plt.imshow(gray_image, cmap="gray")
	plt.title("Original image"), plt.xticks([]), plt.yticks([])
	plt.subplot(142), plt.imshow(fft_box_blur_image, cmap="gray")
	plt.title("Box Blur Image"), plt.xticks([]), plt.yticks([])
	plt.subplot(143), plt.imshow(fft_gaussian_blur_image, cmap="gray")
	plt.title("Gaussian Blur Image"), plt.xticks([]), plt.yticks([])
	plt.subplot(144), plt.imshow(fft_motion_blur_image, cmap="gray")
	plt.title("Motion Blur Image"), plt.xticks([]), plt.yticks([])

	# 4. Ideal Filter plot
	plt.figure(4)
	plt.suptitle("Ideal Filter")
	plt.subplot(121), plt.imshow(gray_image, cmap="gray")
	plt.title("Input Image"), plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(high_pass_filtered_image, cmap="gray")
	plt.title("Ideal filtered image"), plt.xticks([]), plt.yticks([])

	# 5. Image enhancement plot
	plt.figure(5)
	plt.suptitle("Enhancement Process")
	plt.subplot(121), plt.imshow(gray_image, cmap="gray")
	plt.title("Input Image"), plt.xticks([]), plt.yticks([])
	plt.subplot(122), plt.imshow(enhanced_image, cmap="gray")
	plt.title("Enhanced image"), plt.xticks([]), plt.yticks([])

	# 6. Gaussian denoised image Plot
	plt.figure(6)
	plt.suptitle("Gaussian noise Filter")
	plt.subplot(131), plt.imshow(gray_image, cmap="gray")
	plt.title("Input Image"), plt.xticks([]), plt.yticks([])
	plt.subplot(132), plt.imshow(denoised_gaussian_image, cmap="gray")
	plt.title("Denoised image"), plt.xticks([]), plt.yticks([])
	plt.subplot(133), plt.imshow(filtered_gaussian_image, cmap="gray")
	plt.title("Filtered image"), plt.xticks([]), plt.yticks([])

	# 7. Salt and Pepper denoised image Plot
	plt.figure(7)
	plt.suptitle("Salt and Pepper noise Filter")
	plt.subplot(131), plt.imshow(gray_image, cmap="gray")
	plt.title("Input Image"), plt.xticks([]), plt.yticks([])
	plt.subplot(132), plt.imshow(denoised_salt_pepper_image, cmap="gray")
	plt.title("Denoised image"), plt.xticks([]), plt.yticks([])
	plt.subplot(133), plt.imshow(filtered_salt_pepper_image, cmap="gray")
	plt.title("Denoised image"), plt.xticks([]), plt.yticks([])

	plt.show()
