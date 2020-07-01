from setuptools import setup, find_packages

setup(
    name='Regularized-GradientTD',
    url='https://github.com/rlai-lab/Regularized-GradientTD.git',
    author='Andy Patterson',
    author_email='andnpatterson@gmail.com',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        "numpy>=1.19",
        "torch>=1.5.1",
    ],
    version=2.1,
    license='MIT',
)
