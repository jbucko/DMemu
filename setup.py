'''
Created on 15 Februar 2023
@author: Jozef Bucko
Setup script
'''
from setuptools import setup

TBDemu_link = 'https://github.com/jbucko/TBDemu.git'

setup(name='TBDemu',
      version='1.0',
      description='Emulator for predicting effects of two-body decaying dark matter on nonlinear matter power spectrum.',
      url=TBDemu_link,
      author='Jozef Bucko',
      author_email='joz.bucko@gmail.com',
      package_dir = {'TBDemu' : 'src'},
      packages=['TBDemu'],
      package_data={'TBDemu': ['files/*']},
      install_requires=['numpy', 'torch','matplotlib'],
      zip_safe=False,
      include_package_data=True,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      )
