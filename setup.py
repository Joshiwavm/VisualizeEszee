from setuptools import setup, find_packages

setup(
    name='VisualizeEszee',
    version='0.1.0',
    description='SZ cluster modeling and visualization toolkit',
    author='Joshiwavm',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'astropy',
        'pyyaml',
    ],
    include_package_data=True,
    python_requires='>=3.9',
)
