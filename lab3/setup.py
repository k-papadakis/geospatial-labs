from setuptools import setup


setup(
    name='lab3',
    version='0.0.1',
    install_requires=[
        'numpy',
        'matplotlib',
        'torch',
        'torchvision',
        'scikit-learn',
        'pytorch-lightning',
        'torchmetrics',
        'rasterio',
    ],
    packages=['lab3']
)
