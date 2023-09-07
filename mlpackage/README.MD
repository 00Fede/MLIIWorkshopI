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
## Credits
This code was adapted from the following repository https://github.com/rushter/MLAlgorithms. All credits go to the respective creators.
