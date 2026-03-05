from setuptools import setup, find_packages

setup(
    name='VisualizeEszee',
    version='0.1.0',
    description='SZ cluster modeling and visualization toolkit',
    author='Joshiwavm',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'astropy',
        'pyyaml',
        'scipy',
        'corner',
        'reproject',
        'jax',
        'jax_finufft',
    ],
    include_package_data=True,
    python_requires='>=3.10',
)
