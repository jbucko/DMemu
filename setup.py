'''
Created on 15 Februar 2023
@author: Jozef Bucko
Setup script
'''
from setuptools import setup

DMemu_link = 'https://github.com/jbucko/DMemu.git'

setup(name='DMemu',
      version='1.0',
      description='Emulator for predicting effects of two-body decaying dark matter on nonlinear matter power spectrum.',
      url=DMemu_link,
      author='Jozef Bucko',
      author_email='joz.bucko@gmail.com',
      package_dir = {'DMemu' : 'src'},
      packages=['DMemu'],
      package_data={'DMemu': ['files/*']},
      install_requires=['numpy', 'torch','matplotlib'],
      zip_safe=False,
      include_package_data=True,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      )
