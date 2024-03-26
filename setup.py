from setuptools import setup, find_packages

setup(
    name='magicnerf',
    version='0.0.1',
    description='A user-friendly and high-performance implementation of NeRF',
    url='https://github.com/sithu31296/magicnerf',
    author='Sithu Aung',
    author_email='sithu31296@gmail.com',
    license='MIT',
    packages=find_packages(include=['magicnerf']),
    install_requires=[
        'numpy',
        'scipy',
    ]
)