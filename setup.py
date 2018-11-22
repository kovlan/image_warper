from distutils.core import setup, Extension
import numpy

# define the extension module
fastWarper_np = Extension('fastWarper', sources=['./iwarper/c/fastWarper.c'],
                          include_dirs=[numpy.get_include()])

# run the setup
setup(ext_modules=[fastWarper_np])
