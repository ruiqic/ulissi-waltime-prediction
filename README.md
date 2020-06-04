# ulissi-waltime-prediction

Creating and using a KNearestNeighbor regressor to predict the time per step of a given DFT input ase.Atoms object.

Functions of interest are contained in [steptime_regressor.py](steptime_regressor.py)

Dependencies: ulissi [gaspy_regressions](https://github.com/ulissigroup/GASpy_regressions) environment

Additional useful functions: <br>
  * Parsing files to get steptimes: [parsefile_helpers.py](parsefile_helpers.py) 
  * Getting initial atoms object from firework id: [get_initial_atoms.py](get_initial_atoms.py)
