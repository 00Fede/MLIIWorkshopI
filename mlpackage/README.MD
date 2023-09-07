## Project structure

```python
    mlpackage # root
        mymlkit # base package
            metrics # pkg
            svm # pkg
            base # module with basic operations
            linear_models # module of package mymlkit
            pca # module
        tests # tests package
            pca # module to test pca
            svm # module
            ...
```

## Install virtual environment
From folder mlpackage run

``
pipenv install
``

## Update dependencies in requirements.txt
If new dependencies are defined in requirements.txt, run the next command from folder mlpackage to update the environment:

``
pipenv install -r requirements.txt
``

## Test models
In the virtual environment, run

``
python -m tests.svm
``

## To use in jupyter notebooks
Install all the dependencies of Pipfile in python kernel used to run notebook. The following code allows to import the local package:

```python
import os
import sys
module_path = os.path.abspath(os.path.join('.','mlpackage'))
if module_path not in sys.path:
    sys.path.append(module_path)
```

## Credits
This code was adapted from the following repository https://github.com/rushter/MLAlgorithms. All credits go to the respective creators.
