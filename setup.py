from setuptools import setup, find_namespace_packages

install_requires = ['numpy', 'scipy', 'absl-py', 'jax', 'jaxlib']

setup(
    name='fax',
    version='0.0.4',
    packages=find_namespace_packages(
        include=['*', 'fax.*'],
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    url='',
    license='',
    author='Clement Gehring',
    author_email='clement.gehring@gmail.com',
    description='',
    install_requires=install_requires,
)
