import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='psemodel',  
     version='0.1.3',
     author="Andrij David",
     author_email="david@geek.mg",
     description="A Position sequeeze and excitation model (resnet and darknet) for pytorch",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/andrijdavid/Psemodel",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )