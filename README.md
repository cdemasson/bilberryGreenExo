# bilberryGreenExo
Bilberry Technical Exercise for the lead C++ job: Develop a software tool for green segmentation using a Nvidia GPU. 

## Compilation environment
- Intel Core i7-8650U 
- NVIDIA GeForce GTX 1050
- Windows 10 x64

For compatibility with your own Nvidia GPU reference, check the [CUDA ARCH](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards) of your card and replace the correspondong value in the parseGreen.pro file. 

Make sure you have downloaded the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and check that the CUDA_DIR and CUDA_SDK variables in the parseGreen.pro file corresponds to your installation path. 

## How does it work?
You can choose to extract the green color of two different pictures: the puppy Misae or the credit provider Cetelem thanks to the two corresponding buttons on the upper left corner of the window.

The two buttons "Extract Green" and "Extract Green with CUDA" let you choose whether you want to compute the green extraction respectively with the computer processor or its Nvidia graphic card. The elapsed time for this computation is automatically given. A pixel is considered green if the green value is 2 times superior to the red or blue value. 

__Warning:__ For still unknown reasons, the elapsed time for the CUDA computation is really long the first that it is used. Don't hesitate to click a second time to actually observe your graphic card efficiency. 
