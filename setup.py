from distutils.core import setup

setup(
    name='picolo',
    version='0.1.0',
    author='Anna Schneider',
    author_email='annarschneider@gmail.com',
    packages=['picolo', 'config', 'shapes'],
    package_dir={'picolo': 'src/picolo',
                 'config': 'src/picolo/config',
                 'shapes': 'src/picolo/shapes'
                 },
    #scripts=['bin/script.py'],
    #package_data={'mypkg': ['data/*.dat']},
    url='https://github.com/aschn/picolo',
    license='LICENSE.txt',
    description='A library for detecting and classifying local order in spatial data.',
    long_description=open('README.md').read(),
    requires=["matplotlib (>=1.1.1)", "numpy (>=1.6.0)", "scipy (>=0.9.0)"],
    provides=['picolo', 'config', 'shapes']
)
