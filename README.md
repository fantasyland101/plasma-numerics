# WARNING
CURRENT STATE, NOT FULLY TESTED, SEEMS TO WORK

# Setup
First clone the repo then
```
cd plasma-numerics/src
```
dependencies (will package better later):
```
pip install numpy, matplotlib
```

## Run a test by
```
python -m tests.testname
``

## Run the model by

To run the model with parameters from a specific dream data model file (that is not in this repo), put the file at
src/dream_outputs.h5 and then run the model by
```
python ./main.py
```
