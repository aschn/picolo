picolo
=============

A library for detecting and classifying local order in spatial data.

picolo stands for Point-Intensity Classification Of Local Order.


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
