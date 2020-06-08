from setuptools import setup, find_namespace_packages

install_requires = ['numpy', 'scipy', 'absl-py', 'jax', 'jaxlib', 'hypothesis']

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
    author_email='fax-dev@gehring.io',
    description='',
    install_requires=install_requires,
)
