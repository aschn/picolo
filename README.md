picolo
=============

A library for detecting and classifying local order in spatial data. picolo stands for Point-Intensity Classification Of Local Order.

Background
----------

Pattern detection in images is a common task in many fields, including nanoscience.
For instance, there is a need for robust methods that can classify individual membrane proteins as "disordered" or "crystalline," whether the data source is atomic force micrographs, electron micrographs, or molecular simulation trajectories.
picolo meets this need by providing Python tools for extracting features from local neighborhoods in (x,y) data sets (e.g., points picked from micrographs using other software), training and applying statistical classifiers on these features, and computing spatial correlation functions on the labeled data.


Contents
--------

<code>src</code> contains 3 packages: picolo, config, and shapes.
The Trainer, Matcher, and Writer classes in the picolo package are the primary interfaces for building, applying, and analyzing the results of classifiers, respectively.
The config and shapes packages provide various helper classes that are used by picolo.

<code>test</code> contains automated tests that can be run with nosetests.

<code>scripts</code> contains sample scripts, which can be run using files in <code>data</code>.


Installation
------------

    cd picolo
    python setup.py build
    sudo python setup.py install


Documentation and testing
-------------------------

To make documentation with doxypy and LaTeX:

    doxygen Doxyfile
    cd docs/latex
    make refman.pdf

Then open <code>docs/html/index.html</code> or <code>docs/latex/refman.pdf</code>.


To run tests with nose and coverage:

    nosetests --cover-package=picolo --with-coverage


Usage
-----

There are sample scripts in the <code>scripts</code> directory. Call any of the scripts with <code>-h</code> for instructions.


Credits
-------

picolo is written and maintained by Anna Schneider <[annarschneider@gmail.com](mailto:annarschneider+github@gmail.com)>, and was tested on AFM data collected by Bibiana Onoa.


License
-------

picolo is released under the Apache License 2.0.

Copyright 2013 Anna Schneider

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.