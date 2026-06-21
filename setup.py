from setuptools import setup, find_packages

setup(
    name="regime-controller-fx",
    version="0.2.0",
    description="Market regime detection → strategy permission/denial filter for FX/CB trading",
    author="circle_cycle",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "hmm": ["numpy>=1.24", "hmmlearn>=0.3"],
        "all": ["numpy>=1.24", "hmmlearn>=0.3"],
    },
)