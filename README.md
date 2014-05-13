gmpe-smtk
=========

Python and OpenQuake-based Toolkit for Analysis of Strong Motions and Interpretation of GMPEs


dependencies
============

The gmpe-smtk currently requires the following dependencies:


* OpenQuake Hazard Library (oq-hazardlib)
* Numpy (1.6.1 or later) (installed with oq-hazardlib)
* Scipy (0.11.0 or later) (installed with oq-hazardlib)
* Shapely (installed with oq-hazardlib)
* Matplotlib (1.3.x or later)
* h5py (2.2.0)

installation
============

* Windows

Windows users should install the PythonXY software package (https://code.google.com/p/pythonxy/), which will install all of the dependencies except oq-hazardlib 
To install oq-hazardlib it is recommended to install MinGW or Github for Windows.

If using Github for Windows simply open a bash shell, clone the oq-hazardlib
repository using:

>> git clone https://github.com/gem/oq-hazardlib.git

Then type

>> cd oq-hazardlib
>> python setup.py install build --compiler=mingw32

To install the gmpe-smtk simply download the zipped code from the repository,
unzip it to a location of your choice then add the directory path to
the Environment Variables found in:

My Computer -> Properties -> System Properties -> Advanced -> Environment Variables

In the Environment Variables you will see a list of System Variables. Select
"Path" and then "Edit". Then simply add the directory of the gmpe-smtk to the
list of directories.

* OSX/Linux

To install oq-hazardlib simply clone the oq-hazardlib repository into a folder
of your choice.

>> git clone https://github.com/gem/oq-hazardlib.git

Then run

>> cd oq-hazardlib
>> python setup.py install

Matplotlib and h5py can both be installed from the native package managers,
although it is recommended to use pip for this purpose.

To install the gmpe-smtk, clone the code from the repository and then
add the following line to your bash or profile script:

export PYTHONPATH=/path/to/gmpe-smtk/:$PYTHONPATH
