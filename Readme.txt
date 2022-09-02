Material supplementing the article "On Constraints in First-Order Optimization: A View from Non-Smooth Dynamical Systems".

This folder contains the source files that reproduce the experimental results shown in Section 7 of the article. The source files are distributed without any warranty. The entire risk as to the quality and performance of the code is with you.

The code was tested on a Dell Precision Tower 3620 that runs Ubuntu 20.04LTS. We used Python 3.8.5, Cmake 3.16.3, and gcc version 9.3.0 to build the binaries and run the code.

-----------------------------------------------------------------------------------------------

Overview of folders:
pybind11-master		...		pybind11 source
constrainedGD       ...     c++ source files that implement Algorithm 2
constrainedGDlib    ...     Python interface
python				... 	contains the Python code used to run the numerical examples in Section 7
python/Alg2.py		... 	Python implementation of Algorithm 2
python/proxSOR.py   ...     Python implementation of Section 6.1 (projected fixed-point iteration)

-----------------------------------------------------------------------------------------------

Installation:
1) install pybind11: 
cd rootdir/pybind11-master/build
cmake ../
make install
2) build the "constrainedGDlib" Python module (interface Python/C++)
cd rootdir/constrainedGDlib/build
cmake ../
make

The C++ implementation of Algorithm 2 can be accessed in Python through the Python module "constrainedGDlib".


-----------------------------------------------------------------------------------------------

For running the experiments from Section 7.1-7.3 type:
cd rootdir/python
python3 RandomQP/runRandomQPCpp.py
python3 ThrustRegion/runThrustRegionCpp.py
python3 nuSVM/runNuSVMCpp.py


For running the Catenary example from Section 7.4 type:
cd rootdir/python/Catenary
python3 main.py


