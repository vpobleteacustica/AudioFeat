import pandas as pd
import os
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt 
import librosa
import torch
from io import BytesIO
from PIL import Image
import IPython.display as ipd
from IPython.display import Audio
from IPython import display
import IPython.display
import torchvision
import kornia
from kornia.color import rgb_to_grayscale, rgba_to_rgb
from kornia.color import rgb_to_grayscale, rgba_to_rgb
from kornia.color.gray import grayscale_to_rgb
from kornia.color.rgb import rgb_to_rgba
from kornia.core import Device, Tensor
from kornia.core.check import KORNIA_CHECK
from typing import Optional
from kornia.color.rgb import bgr_to_rgb
from kornia.core import Module, Tensor, concatenate
from kornia.core.check import KORNIA_CHECK_IS_TENSOR

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (dB)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    
def process(sndf1):
    
    Nfft = 1024
    blocksize = 4096
    channelChoice = 0

    psd_Aves1 = []
    psd_Matrix1 = []
    f_values1 = []
    cnt1 = 0
    while sndf1.tell() < len(sndf1) - blocksize:
        data1 = sndf1.buffer_read(blocksize, dtype='int16')
        if sndf1.channels == 2:
            if channelChoice == -1:
                ch0 = np.average(np.abs(np.frombuffer(data1, dtype='int16')[0::2]))
                ch1 = np.average(np.abs(np.frombuffer(data1, dtype='int16')[1::2]))
                if ch0 > ch1:
                    channelChoice = 0
                else:
                    channelChoice = 1
            npData1 = np.frombuffer(data1, dtype='int16')[channelChoice::2]
        else:
            npData1 = np.frombuffer(data1, dtype='int16')
        f_values1, psd_values1 = welch(npData1, fs=sndf1.samplerate, nfft=Nfft, scaling='spectrum')
            
        if psd_Aves1 == []:
            psd_Aves1 = psd_values1
            psd_Matrix1 = psd_values1
        else:
            psd_Aves1 += psd_values1
            psd_Matrix1 = np.append(psd_Matrix1, psd_values1, axis = 0)
        cnt1 += 1

    dBaves1 = 10 * np.log10(psd_Aves1/cnt1)
    
    return f_values1, dBaves1

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate
  
  figure, axes = plt.subplots(num_channels, 1)
    
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False) 

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def plot_fft_custom(time_vector, x, filename, sample_rate, freqs, X):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8)) 
    fig.subplots_adjust(hspace=.5)

    plt.rc('xtick', labelsize=18) 
    plt.rc('ytick', labelsize=18)

    ax1.plot(time_vector, x, 'grey', label = '$x(t)$', linewidth=2)
    ax1.set_xlabel('time (seconds)', fontsize=18)
    ax1.set_ylabel('amplitude', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.rc('legend',**{'fontsize':16})
    ax1.legend()
    ax1.set_title('Continuous time signal: ' + str(filename), fontweight="bold", size=16)
    ax1.grid()

    ax2.plot(freqs, X, 'k', label = '$|X(f)|$', linewidth=2)
    ax2.set_xlabel('frequency (Hz)', fontsize=18)
    ax2.set_ylabel('$|X(f)|$', fontsize=18)
    ax2.legend()
    ax2.set_title('Bilateral magnitude spectrum between: ' + str(-int(sample_rate/2)) + ' Hz ' + ' to ' + str(int(sample_rate/2)) + ' Hz', fontweight="bold", size=16)
    ax2.grid()
    
    plt.show(block=False)

def plot_image(tensor):
    plt.figure()
    plt.imshow(tensor.numpy().transpose(1, 2, 0))
    plt.show()
    
def show_grayscale_image(tensor):
    
    f = BytesIO()
    a = np.uint8(tensor.mul(255).numpy()) 
    Image.fromarray(a).save(f, 'png')
    IPython.display.display(IPython.display.Image(data = f.getvalue()))
    
def plot_grayscale_image(tensor):
    plt.figure()
    plt.imshow(tensor.numpy(), cmap = 'gray')
    plt.show() 

def hflip(input: torch.Tensor) -> torch.Tensor:
    return torch.flip(input, [-1])


def vflip(input: torch.Tensor) -> torch.Tensor:
    return torch.flip(input, [-2])


def rot180(input: torch.Tensor) -> torch.Tensor:
    return torch.flip(input, [-2, -1])


def imshow(input: torch.Tensor):
    out: torch.Tensor = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np: np.ndarray = kornia.tensor_to_image(out)
    plt.imshow(out_np)
    plt.axis('off')
    plt.show()

def grayscale_to_rgb(image: Tensor) -> Tensor:
    r"""Convert a grayscale image to RGB version of image.

    .. image:: _static/img/grayscale_to_rgb.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: grayscale image to be converted to RGB with shape :math:`(*,1,H,W)`.

    Returns:
        RGB version of the image with shape :math:`(*,3,H,W)`.

    Example:
        >>> input = torch.randn(2, 1, 4, 5)
        >>> gray = grayscale_to_rgb(input) # 2x3x4x5
    """
    KORNIA_CHECK_IS_TENSOR(image)

    if image.dim() < 3 or image.size(-3) != 1:
        raise ValueError(f"Input size must have a shape of (*, 1, H, W). " f"Got {image.shape}.")

    return concatenate([image, image, image], -3)



def rgb_to_grayscale(image: Tensor, rgb_weights: Optional[Tensor] = None) -> Tensor:
    r"""Convert a RGB image to grayscale version of image.

    .. image:: _static/img/rgb_to_grayscale.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB image to be converted to grayscale with shape :math:`(*,3,H,W)`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = rgb_to_grayscale(input) # 2x1x4x5
    """
    KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    if rgb_weights is None:
        # 8 bit images
        if image.dtype == torch.uint8:
            rgb_weights = torch.tensor([76, 150, 29], device=image.device, dtype=torch.uint8)
        # floating point images
        elif image.dtype in (torch.float16, torch.float32, torch.float64):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")
    else:
        # is tensor that we make sure is in the same device/dtype
        rgb_weights = rgb_weights.to(image)

    # unpack the color image channels with RGB order
    r: Tensor = image[..., 0:1, :, :]
    g: Tensor = image[..., 1:2, :, :]
    b: Tensor = image[..., 2:3, :, :]

    w_r, w_g, w_b = rgb_weights.unbind()
    return w_r * r + w_g * g + w_b * b



def bgr_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to grayscale.

    The image data is assumed to be in the range of (0, 1). First flips to RGB, then converts.

    Args:
        image: BGR image to be converted to grayscale with shape :math:`(*,3,H,W)`.

    Returns:
        grayscale version of the image with shape :math:`(*,1,H,W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = bgr_to_grayscale(input) # 2x1x4x5
    """
    KORNIA_CHECK_IS_TENSOR(image)

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    image_rgb: Tensor = bgr_to_rgb(image)
    return rgb_to_grayscale(image_rgb)



class GrayscaleToRgb(Module):
    r"""Module to convert a grayscale image to RGB version of image.

    The image data is assumed to be in the range of (0, 1).

    Shape:
        - image: :math:`(*, 1, H, W)`
        - output: :math:`(*, 3, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Example:
        >>> input = torch.rand(2, 1, 4, 5)
        >>> rgb = GrayscaleToRgb()
        >>> output = rgb(input)  # 2x3x4x5
    """

    def forward(self, image: Tensor) -> Tensor:
        return grayscale_to_rgb(image)


class RgbToGrayscale(Module):
    r"""Module to convert a RGB image to grayscale version of image.

    The image data is assumed to be in the range of (0, 1).

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = RgbToGrayscale()
        >>> output = gray(input)  # 2x1x4x5
    """

    def __init__(self, rgb_weights: Optional[Tensor] = None) -> None:
        super().__init__()
        self.rgb_weights = rgb_weights

    def forward(self, image: Tensor) -> Tensor:
        return rgb_to_grayscale(image, rgb_weights=self.rgb_weights)



class BgrToGrayscale(Module):
    r"""Module to convert a BGR image to grayscale version of image.

    The image data is assumed to be in the range of (0, 1). First flips to RGB, then converts.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 1, H, W)`

    reference:
        https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> gray = BgrToGrayscale()
        >>> output = gray(input)  # 2x1x4x5
    """

def forward(self, image: Tensor) -> Tensor:
        return bgr_to_grayscale(image)