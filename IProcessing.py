import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

def convert_to_gray(image):

  width, height, dimensions = image.shape
  gray_image = np.zeros((height,width), np.uint8)

  b,g,r = cv2.split(image)

  w = np.array([[[ 0.3, 0.55,  0.15]]])
  gray_image = cv2.convertScaleAbs(np.sum(image*w, axis=2))

  return gray_image

def blur_box_convolve(image,size = 3):
  pad = (size-1)//2
  kernel = np.ones((size,size), np.float32)/9

  image = cv2.copyMakeBorder(image,pad,pad,pad,pad, cv2.BORDER_REPLICATE)
  image = np.uint8(signal.convolve2d(image, kernel))
  
  return image

def blur_gaussian_convolve(image, size=3):
  pad = (size-1)//2
  if size==3:
    kernel = np.array([[1,2,1],
                      [2,4,2],
                      [1,2,1]], np.float32)/16

  elif size==5:
    kernel = np.array([[1,4,7,4,1],
                       [4,16,26,16,4],
                       [7,26,41,26,7],
                       [4,16,26,16,4],
                       [1,4,7,4,1]], np.float32)/273

  else:
    kernel = np.array([[0,0,1,2,1,0,0],
                       [0,3,13,22,13,3,0],
                       [1,13,59,97,59,13,1],
                       [2,22,97,159,97,22,2],
                       [1,13,59,97,59,13,1],
                       [0,3,13,22,13,3,0],
                       [0,0,1,2,1,0,0]], np.float32)/1003

  image = cv2.copyMakeBorder(image,pad,pad,pad,pad, cv2.BORDER_REPLICATE)
  image = np.uint8(signal.convolve2d(image, kernel))

  return image

def blur_motion_convolve(image):
  pad = 2
  kernel = np.array([[0,0,1,0,0],
                     [0,0,1,0,0],
                     [0,0,1,0,0],
                     [0,0,1,0,0],
                     [0,0,1,0,0]] ,np.float32)/9

  image = cv2.copyMakeBorder(image,pad,pad,pad,pad, cv2.BORDER_REPLICATE)
  image = np.uint8(signal.convolve2d(image, kernel))
  
  return image

def blur_box_FFT(image,size = 3):
  pad = (size-1)//2
  Hi, Wi = image.shape
  
  # Image Border
  image = cv2.copyMakeBorder(image,pad,pad,pad,pad, cv2.BORDER_REPLICATE)

  # Blur Kernel
  kernel = np.ones((size,size), np.float32)/9
  Hk, Wk = kernel.shape

  # Pad Kernel
  sz = (image.shape[0] - kernel.shape[0], image.shape[1] - kernel.shape[1])
  kernel = np.pad(kernel, (((sz[0]+1)//2, sz[0]//2), ((sz[1]+1)//2, sz[1]//2)), 'constant')

  # Shifts kernel origin to 1st image
  # pixel(top-left corner)
  kernel= np.fft.ifftshift(kernel)

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

if __name__ == '__main__':

  # Read Image
  original_image=cv2.imread("me.JPG", -1)
  original_image=cv2.resize(original_image, (300,400))

  # Convert do Gray
  gray_image = convert_to_gray(original_image) 

  # Gets images's spectrum
  image_spectrum = cv2.dft(np.float32(gray_image), 
                            flags=cv2.DFT_COMPLEX_OUTPUT)

  image_spectrum = np.fft.fftshift(image_spectrum)

  original_magnitude_spectrum = 20*np.log(
                                  cv2.magnitude(
                                    image_spectrum[:,:,0],
                                    image_spectrum[:,:,1]))


  # Image convolve Blurring
  box_blur_image = blur_box_convolve(gray_image)
  gaussian_blur_image = blur_gaussian_convolve(gray_image, 7)
  motion_blur_image = blur_motion_convolve(gray_image)
  
  # Image FFT Blurring
  fft_box_blur_image = blur_box_FFT(gray_image)


  plt.figure(1)
  plt.subplot(121), plt.imshow(gray_image, cmap = 'gray')
  plt.title('Original image'), plt.xticks([]), plt.yticks([])
  plt.subplot(122),plt.imshow(box_blur_image, cmap = 'gray')
  plt.title('Box Blur Image'), plt.xticks([]), plt.yticks([])

  plt.figure(2)
  plt.subplot(121),plt.imshow(gray_image, cmap = 'gray')
  plt.title('Input Image'), plt.xticks([]), plt.yticks([])
  plt.subplot(122),plt.imshow(gaussian_blur_image, cmap = 'gray')
  plt.title('Gaussian Blur Image'), plt.xticks([]), plt.yticks([])

  plt.figure(3)
  plt.subplot(121),plt.imshow(gray_image, cmap = 'gray')
  plt.title('Input Image'), plt.xticks([]), plt.yticks([])
  plt.subplot(122),plt.imshow(motion_blur_image, cmap = 'gray')
  plt.title('Motion Blur Image'), plt.xticks([]), plt.yticks([])
  
  plt.figure(4)
  plt.subplot(121),plt.imshow(gray_image, cmap = 'gray')
  plt.title('Input Image'), plt.xticks([]), plt.yticks([])
  plt.subplot(122),plt.imshow(original_magnitude_spectrum, cmap = 'gray')
  plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

  plt.figure(5)
  plt.subplot(121),plt.imshow(gray_image, cmap = 'gray')
  plt.title('Input Image'), plt.xticks([]), plt.yticks([])
  plt.subplot(122),plt.imshow(fft_box_blur_image, cmap = 'gray')
  plt.title('FFT blur box image'), plt.xticks([]), plt.yticks([])

  plt.show()