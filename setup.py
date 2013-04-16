from distutils.core import setup

setup(
    name='ShapeMatchLib',
    version='0.1.0',
    author='Anna Schneider',
    author_email='annarschneider@gmail.com',
    packages=['shapematchlib', 'config'],
    package_dir={'shapematchlib': 'src/shapematchlib',
                 'config': 'src/shapematchlib/config'},
   # scripts=['bin/script.py'],
    #package_data={'mypkg': ['data/*.dat']},
    url='https://github.com/aschn/shapematchlib',
    license='LICENSE.txt',
    description='A library for detecting and classifying local order in spatial data.',
    long_description=open('README.md').read(),
    install_requires=[
        "matplotlib >= 1.1.1",
        "numpy >= 1.6.0",
        "scipy >= 0.9.0",
        "sklearn >= 0.12.1",
    ],
)
