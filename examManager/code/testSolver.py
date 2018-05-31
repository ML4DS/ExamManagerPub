# This script tests the behavior of the exam solver for a given exam database
# project.
# To do so, it proceeds by generating all posible datasets for the student and
# solving the sequence of all exam questions available in the database.
#
# Usage:
#
#     python testSolver --p solverName
#
# where solverName y the tag identifying the solver to call. Available tags are
#
#     B12
#     B3
#     TD
#
# Version: 1.0, JCS (march, 2018)

import argparse
from scipy.io import loadmat


# ####################
# Read input arguments
# ####################

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default=None)
args = parser.parse_args()

# Conditional imports
if args.p == 'B12':
    from lib.dbSolverB12 import computeAll
    from lib.examProjectB12 import generateData
elif args.p == 'B3':
    from lib.dbSolverB3 import computeAll
    from lib.examProjectB3 import generateData
elif args.p == 'TD':
    from lib.dbSolverTD import computeAll
    from lib.examProjectTD import generateData
elif args.p is None:
    print("\nUnspecified solver. Select the solver to test with ")
    print("     python --p solverName")
    print("Available values of solverName are B12, B3 and TD.\n")
    exit()
else:
    print("\nWrong solver name")
    print("Available values of solverName are B12, B3 and TD.\n")
    exit()

# ##################
# Full db resolution
# ##################

# Solving all questions for all possible dataset.
print(36 * '-' + '\nTesting all possible hash values')
n_max = 1000
for nia in range(n_max):

    print('-- User {0} out of {1}\r'.format(nia, n_max), end='', flush=True)
    data = generateData(nia)
    results, scoring = computeAll(data)

print('--- Systematic Test OK: no errors detected.')

# #########################
# Testing a saved data file
# #########################

# Test and show single mat file
data = generateData(0)

print('\n\n' + 36 * '-')
print('--- Example from arbitrary data file')
results, scoring = computeAll(data)

print('Results = ')
print(results)
for item in results:

    if hasattr(results[item], '__len__') and len(results[item]) > 50:
        print(item + ' = shape {0}'.format(results[item].shape))
    else:
        print(item + ' = {0}'.format(results[item]))

print('Scores = ')
for item in scoring:

    print('Score for {0}: {1}'.format(item, scoring[item]))