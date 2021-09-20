from setuptools import setup, find_packages

p = find_packages()
print(p)
setup(
    name='thesis',
    version='0.1.0',
    packages=p
)
