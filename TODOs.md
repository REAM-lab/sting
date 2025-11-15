# TODOs

## General
- Update export to matlab in the system class? 
- Should we replace `linalg.inv` with `linalg.solve`?
- Dataclass for component connection matrices?
- Variable/State descriptions
    - Can we put this in the globally scoped atrrs of each class?
- Update gen models to new state-space model
- Update naming to camelcase classes 
- Auto detect input files in system start up when reading from CSVs

## Adam Subtasks
1. Testing
    - Add unit tests
        - StateSpaceModels
        - Variables
    - Validation
        - Update CSVs from MATLAB to Python
        - Define test inputs
        - Save EMT output files

2. Model reduction
    - Finish SSM class
    - Add gramians
    - Add intraconnect

3. Other
 - Finish ComponentConnections class


## Paul Subtasks