# TODOs

## General
- Do we need export to matlab in the system class? Or is it sufficient to use it just in the composite model class?
    - Also this will need to be updated with the new implementation of `CompositeModel`
- Should we replace `linalg.inv` with `linalg.solve`.
- Option to run string from a given directory, not just cwd?

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
    - Finish composite model class
    - Finish SSM class
    - Add gramians
    - Add intraconnect

3. Other
 - Finish ComponentConnections class


## Paul Subtasks