# IRGPUA PROJECT

## Project presentation :

### Introduction

AMDIA company has a mission for you :</br>
The internet is broken → All images are corrupted and are unusable

After some deep analysis performed by the LRE labs, the issue was found :</br>
A hacker named Riri has introduced garbage data at the entry of Internet pipelines

The corruption is as follows :
* Garbage data in the form of lines of “-27” have been introduced in the image</br>
* Pixels got their values modified : 0 : +1 ; 1 : -5 ; 2 : +3 ; 3 : -8 ; 4 : +1 ; 5 : -5; 6 : +3; 7 : -8…</br>
* All images have awful colors and should be histogram equalized

Engineers from AMDIA have found a solution but it is super slow (CPU folk’s issues)</br>
Your goal is to fix the issue in real-time

AMDIA knows you will succeed in the mission and has a final task for you :</br>
Once all images from the pipeline are cleaned up, they want to do some statistics</br>
You should first compute each image’s total sum of all its pixel values</br>
Then sort the resulting images based on this total sum (optional)</br>
Of course, all of this should be done in the fastest way possible

They will provide you with their CPU code to start and test your implementation

### Advices

Almost all of the algorithms / patterns used for the project will be practiced in TP</br>
Do the TP and ask questions to have the best base to start from</br>
Implementing a working (aka slow) version of the remaining algorithms / patterns should not be too complicated</br>
Once and only once your pipeline is working, you should start optimizing</br>
You will be working in groups of 3. Split tasks (parallel work) and communicate to share ideas about optimization ideas. You should think together but I advise you to code separately so that you all learn and have time to finish the project.</br>
If you have any questions : nblin@lrde.epita.fr

### Expectations

* Have a working pipeline (fix images + statistics; sorting on GPU is optional)
* Have optimized algorithms / patterns
* Have Nsight Compute & Systems reports (screenshots) to show your performance analysis
* Have a second working industrial (fast) version using CUB, Thrust… (you should minimize the use of hand-made kernel as much as possible for this version)
* Have a presentation for your defense explaining how you programmed, analyzed, and optimized each algorithm / pattern and how you used libraries for your industrial version
* I am not expecting any report, just slides with explanation and Nsight screenshots
* I prefer having algorithms extra optimized rather than great slides ! Don’t spend too much time on it

### About the code

Path for the images has been hardcoded for you and should be working on the OpenStack.

If you still want to download the "images" folder : https://cloud.lrde.epita.fr/s/srHKqQkiz87CjNP
(Feel free to then change the "images" path in main.cu if you have your downloaded the folder)

The project layout is simple :
* An "images" folder containing the broken images you need to fix (it is in the afs)
* "main.cu" contains the code used to : Load the images from the pipeline, fix the images on CPU & compute the statistics and writing the results
* "fix_cpu.hh/pp" contains the code that fixes images on CPU
* "image.hh" and "pipeline.hh" simply contains the code to handle both images and pipeline

I advise you to work with only one image at first to avoid file loading time. Moreover it will be easier to debug if you always work with the same image.

You can modify all the code as you wish but I have highlighted the code with "TODO" where you should act.
You should not have to modify the pipeline.h and image.h files. You can but make sure the baseline CPU code still works **exactly** the same.
You **must not** modify the fix_cpu files.
You **must change** the main.cu to fix images on both CPU & GPU and check your results.

Feel free to create more files, classes...

PS : I know some images look weird after the CPU histogram equalization, it is "normal"
(To be fair, not all the images are actually good candidates for the histogram equalization
But for the exercise, having you do an optimized histogram is cool :)

### Requirements to build

* [Cuda Toolkit](https://developer.nvidia.com/cuda-downloads)
* C++ compiler ([g++](https://gcc.gnu.org/) for linux,  [MSVC](https://visualstudio.microsoft.com/downloads/) for Windows)
* [GPU supported by CUDA](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
* [CMake](https://cmake.org/download/)

### Build

- To build, execute the following commands :

```bash
mkdir build && cd build
cmake ..
make -j
```

* By default the program **will run in release** when it's inside a `build` or `build_release` folder. To build in **debug**, build the projet inside a `build_debug` folder.

### Run :

```bash
cd build
./bin/main
```
