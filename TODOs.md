# TODOs

## General
- Should we replace `linalg.inv` with transpose?
- Dataclass for component connection matrices?
- Variable/State descriptions
    - Can we put this in the globally scoped atrrs of each class?
- Can we create a silent mode? This would be helpful for testing

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

## Paul Subtasks