import setuptools

install_requires = ['numpy', 'scipy', 'absl-py', 'jax', 'jaxlib', 'hypothesis']

with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
    name='jax-fixedpoint',
    version='0.0.4',
    description='Implicit and competitive differentiation in JAX.',
    packages=setuptools.find_namespace_packages(
        include=['*', 'fax.*'],
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    url='',
    license='MIT License',
    author='Clement Gehring',
    author_email='fax-dev@gehring.io',
    long_description=long_description.strip(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        "License :: OSI Approved :: MIT License",

        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",

        "Programming Language :: C++",
        "Programming Language :: Python :: 3",

        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.5",
)
