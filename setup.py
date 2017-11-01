from setuptools import setup, find_packages

setup(
    name='util_bundle',
    version='0.1',
    keywords='util',
    url='https://github.com/Fairly/util_bundle.git',
    author='Shanzhuo Zhang',
    author_email='shanzhuo.zhang@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['numpy', 'matplotlib', 'docopt', 'schema']
)
