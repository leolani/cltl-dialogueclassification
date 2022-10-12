from setuptools import setup, find_namespace_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()


setup(
    name='cltl.dialogueclassification',
    version=version,
    package_dir={'': 'src'},
    packages=find_namespace_packages(include=['cltl.*', 'cltl_service.*'], where='src'),
    data_files=[('VERSION', ['VERSION'])],
    url="https://github.com/leolani/cltl-dialogueclassification",
    license='MIT License',
    author='CLTL',
    author_email='t.baier@vu.nl',
    description='Template component for Leolani',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
    install_requires=[],
    extras_require={
        "impl": [
            "transformers",
            'torch',
            'numpy'
        ],
        "service": [
            "cltl.combot",
            "emissor",
        ]}
)