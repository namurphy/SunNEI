from __future__ import print_function
from sunnei import read_atomic_data, create_ChargeStates_dictionary

def test_read_atomic_data():
    print("Running test_read_atomic_data")
    AtomicData = read_atomic_data(screen_output=True)
    print("Completed test_read_atomic_data")

def test_create_ChargeStates_dictionary():

    elements = ['H', 'He', 'Fe']

    AtomicData = read_atomic_data(elements)

    CSD = create_ChargeStates_dictionary(elements, 2.5879e8, AtomicData)

    for element in elements:
        print()
        print(element+':')
        print(CSD[element])

    
    
