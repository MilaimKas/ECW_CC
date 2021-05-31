
from setuptools import setup

setup(
    name='ECW_CC',
    version='0.1.0',
    description='Experimentally Constrained Wave function Coupled Cluster',
    url='https://github.com/MilaimKas/ECW_CC',
    author='Milaim Kas',
    author_email='milaim.kas@gmail.com',
    license='MIT License',
    packages=['ECW_CC'],
    install_requires=['PySCF',
                      'numpy',
                      'matplotlib',
                      'tabulate',
                      'scipy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science',
        'License :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.9'
    ],
)