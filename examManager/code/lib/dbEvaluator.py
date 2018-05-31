# This script contains a set of functions solving all questions stated in the
# filtering exam database

from __future__ import print_function
import numpy as np
from numbers import Number
import copy


def computeScore(v_test, v_ref, scoring=None):
    """
    Computes the score for a variable v_test with respect to the reference
    value in v_ref
    """

    if scoring is None:
        s = 1
        tol = 1e-4
        scoring = [(score, tol)]

    if v_test is None:
        # No student response
        score = 0

    elif np.any([hasattr(x, '__len__') for x in np.array(v_test).flatten()]):
        # The student response
        print("Warning: the student response has a non-standard structure.")
        print("         It is likely an array containing mixed type elements")
        score = 0

    elif np.ndim(np.array(v_test)) == 0 and np.isnan(v_test):
        # No student response
        score = 0

    elif (np.ndim(np.array(v_test)) > 0 and
          np.all([np.isnan(x) for x in np.array(v_test)])):
        # No student response
        score = 0

    elif isinstance(v_ref, str):
        # String comparisons
        if isinstance(v_test, str):
            score = v_ref == v_test
        else:
            score = 0

    else:

        # Evaluation of numerical or vectorial responses
        v_ref_flat = np.array(v_ref).flatten()
        v_test_flat = np.array(v_test).flatten()

        if np.array_equal(v_test_flat.shape, v_ref_flat.shape):

            # Compute score as a function of the tolerance values
            err = np.mean(np.abs(v_ref_flat - v_test_flat))
            score = max(map(lambda x: x[0] * (err <= x[1]), scoring))

        else:

            score = 0

    return score


def normalizeResponse(resp, sol):
    '''
    Normalizes the response variable in resp to the shape of the solution
    variable in sol.
    '''

    # Get and adapt student variables that can be used as intermedate
    # variables to solve further questions.
    if (hasattr(resp, '__len__') and not isinstance(resp, str) and
            hasattr(sol, '__len__') and not isinstance(sol, str)):

        # This is to convert resp to numpy array just in case it is a list
        resp_norm = np.array(resp)

        # If the student response has the same size than the true solution,
        # reshape the student response to the shape of the true solution.
        if resp_norm.size == sol.size and not np.array_equal(resp_norm.shape,
                                                             sol.shape):
            resp_norm = np.reshape(np.array(resp_norm), sol.shape)
        else:
            # If resp and resp_norm have not the same numver of element, there
            # is nothing to do.
            resp_norm = resp
    else:
        resp_norm = resp

    return resp_norm


def isValidResponse(resp, sol):
    ''' Verifies if resp is a response of the same kind than sol, in such a
        way that it can be used as an intermediate solution variable

        This code assumes that a vector variable is a valid response if it has
        the same number of components than the solution variable. Thus, we are
        assuming that the question of the exam is requesting a variable with a
        fixed size.
    '''

    out = False

    if isinstance(resp, str) and isinstance(sol, str):
        out = True

    elif (hasattr(resp, '__len__') and hasattr(sol, '__len__') and
          resp.size == sol.size):
        out = True

    elif isinstance(resp, Number) and isinstance(sol, Number):
        out = True

    return out


def solveExam(questions, data, solveQuestion):
    '''
    Solves the exam given by a list of questions and a set of data variables.

    Args:

        questions: List of strings specifying the questions in the database to
                   solve.
        data:      Dictionary with the primary data provided to the student.
                   Each entry contains one variable
    '''

    # This is the solution based on the original data
    # and the solutions to the previous questions.
    state = {}
    solution = {}
    scoring_ex = {}
    for q in questions:

        # Solve question q
        # print('    Solving question {}'.format(q))
        sol_q, scoring, state, pointer, transform = solveQuestion(
            q, data, state)
        # print('        solved variables {}'.format(sol_q.keys()))

        # Add solution of this question to the exam solution
        solution = {**solution, **sol_q}
        scoring_ex = {**scoring_ex, **scoring}

        # Add new state variables from the variables named in pointer
        for var, var2 in pointer.items():
            # Warning: this should be updatetd in dbSolverB3.py
            if var2 in data:
                state[var] = data[var2]
            elif var2 in state:
                print(("Warning: variable {} was found unexpectedly in the " +
                       "state dictinary. Maybe it is ok, but " +
                       "maybe you should check it").format(pointer[var]))
            elif var2 in solution:
                # Warning: this should be updatetd in dbSolverB3.py
                state[var] = solution[var2]

    return solution, scoring_ex


def evaluateExam(questions, data, response, solveQuestion):

    # ########################
    # Compute correct solution
    # ########################

    solution_ex, scoring_ex = solveExam(questions, data, solveQuestion)

    # ###########################
    # Normalize student responses
    # ###########################

    # Get and adapt student variables that can be used as intermedate
    # variables to solve further questions.
    response_norm = {}
    for var in solution_ex:
        response_norm[var] = normalizeResponse(response[var], solution_ex[var])

    # ##############################
    # Compute student-based solution
    # ##############################

    # Student-based solution. Each solution is computed based on the previous
    # student solutions
    state = {}
    solution_exS = {}

    for q in questions:

        # Solve question q
        sol_q, scoring, state, pointer, transform = solveQuestion(
            q, data, state)

        # Add solution of this question to the exam solution
        solution_exS = {**solution_exS, **sol_q}

        # Get and adapt student variables that can be used as intermedate
        # variables to solve further questions.
        for var in sol_q:
            if isValidResponse(response_norm[var], sol_q[var]):
                state[var] = response_norm[var]

        # Add new state variables from the variables named in pointer
        for var, var2 in pointer.items():
            if var2 in data:
                # Var is pointing to a variable exising in the original data
                state[var] = data[var2]
            elif var2 in state:
                # Var is pointing to a variable exising in the state
                state[var] = state[var2]
            # Var is pointing to a variable exising in the current
            # solutions
            elif isValidResponse(response_norm[var2], solution_ex[var2]):
                state[var] = response_norm[var2]
            else:
                state[var] = solution_ex[var2]

    # ####################
    # Compute final report
    # ####################

    examReport = {}
    finalscore = 0
    nvar = 0

    for var in solution_ex:

        if var in scoring_ex and var in response_norm:
            score = computeScore(response_norm[var], solution_ex[var],
                                 scoring_ex[var])
            weight = 1
            scoreS = computeScore(response_norm[var], solution_exS[var],
                                  scoring_ex[var])
            weightS = 0.9
            if var in transform:
                weightT = 0.8
                try:
                    z = transform[var](response_norm[var])
                    z_norm = normalizeResponse(z, solution_ex[var])
                    scoreT = computeScore(z_norm, solution_exS[var],
                                          scoring_ex[var])
                except:
                    scoreT = 0
            else:
                scoreT = 0
                weightT = 0

            examReport[var] = {}

            resp = response_norm[var]
            sol = solution_ex[var]

            # Check if the dimension of the response variable coincides with
            # that of the solution
            examReport[var]['Dim'] = 'OK'    # By default, no message
            if isinstance(resp, str) and not isinstance(sol, str):
                examReport[var]['Dim'] = 'Error'
            elif hasattr(sol, '__len__') and hasattr(resp, '__len__'):
                if sol.size != resp.size:
                    examReport[var]['Dim'] = 'Error'
            elif resp is None or np.any(np.isnan(resp)):
                examReport[var]['Dim'] = 'No data'
            elif hasattr(sol, '__len__'):
                if sol.size > 1:
                    examReport[var]['Dim'] = 'Error'
            elif hasattr(resp, '__len__'):
                if resp.size > 1:
                    examReport[var]['Dim'] = 'Error'

            # I should revise the variable names, because score is assigned to
            # 'value' and score * weight is assigned to 'score'
            sw = score * weight
            swS = scoreS * weightS
            swT = scoreT * weightT
            if sw >= swS and sw >= swT:
                examReport[var]['w'] = weight
                examReport[var]['s'] = score
                examReport[var]['w·s'] = sw
            elif swS >= swT:
                examReport[var]['w'] = weightS
                examReport[var]['s'] = scoreS
                examReport[var]['w·s'] = swS
            else:
                examReport[var]['w'] = weightT
                examReport[var]['s'] = scoreT
                examReport[var]['w·s'] = swT

            finalscore += examReport[var]['w·s']
            nvar += 1

    totalscore = 10 * finalscore / nvar
    examReport['Exam'] = {'Score': totalscore}

    return examReport


def evaluateExamOld(questions, data, response, solveQuestion):

    # ########################
    # Compute correct solution
    # ########################

    solution_ex, scoring_ex = solveExam(questions, data, solveQuestion)

    # ###########################
    # Normalize student responses
    # ###########################

    # Get and adapt student variables that can be used as intermedate
    # variables to solve further questions.
    response_norm = {}
    for var in solution_ex:
        response_norm[var] = normalizeResponse(response[var], solution_ex[var])

    # ##############################
    # Compute student-based solution
    # ##############################

    # Student-based solution. Each solution is computed based on the previous
    # student solutions
    data2 = {}
    solution_exS = {}

    for q in questions:

        # Solve question q
        sol_q, state, transform, scoring = solveQuestion(q, data, data2)

        # Add solution of this question to the exam solution
        solution_exS = {**solution_exS, **sol_q}

        # Get and adapt student variables that can be used as intermedate
        # variables to solve further questions.
        for var in sol_q:
            if isValidResponse(response_norm[var], sol_q[var]):
                data2[var] = response_norm[var]
            elif var in data2:
                # Remove the existing valuesin data2, because they should
                # correspond to another question
                del data2[var]

        # Add additional variables from state
        for var in state:
            if type(state[var]) == str:
                var2 = state[var]

                if isValidResponse(response_norm[var2], solution_ex[var2]):
                    data2[var] = response_norm[var2]
                elif var in data2:
                    # Remove the existing values in data2, because they should
                    # correspond to another question
                    print("WARNING: Evaluating question {}".format(q))
                    print("         Removing older response to {}".format(var))
                    del data2[var]

            else:
                data2[var] = state[var]

    # ####################
    # Compute final report
    # ####################

    examReport = {}
    finalscore = 0

    for var in solution_ex:

        score = computeScore(response_norm[var], solution_ex[var],
                             scoring_ex[var])
        weight = 1
        scoreS = computeScore(response_norm[var], solution_exS[var],
                              scoring_ex[var])
        weightS = 0.9
        if var in transform:
            weightT = 0.8
            try:
                z = transform[var](response_norm[var])
                z_norm = normalizeResponse(z, solution_ex[var])
                scoreT = computeScore(z_norm, solution_exS[var],
                                      scoring_ex[var])
            except:
                scoreT = 0
        else:
            scoreT = 0
            weightT = 0

        examReport[var] = {}

        resp = response_norm[var]
        sol = solution_ex[var]

        # Check if the dimension of the response variable coincides with that
        # of the solution
        examReport[var]['Dim'] = 'OK'    # By default, no message
        if hasattr(sol, '__len__') and hasattr(resp, '__len__'):
            if sol.size != resp.size:
                examReport[var]['Dim'] = 'Error'
        elif resp is None or np.any(np.isnan(resp)):
            examReport[var]['Dim'] = 'No data'
        elif hasattr(sol, '__len__'):
            if sol.size > 1:
                examReport[var]['Dim'] = 'Error'
        elif hasattr(resp, '__len__'):
            if resp.size > 1:
                examReport[var]['Dim'] = 'Error'

        # I should revise the variable names, because score is assigned to
        # 'value' and score * weight is assigned to 'score'
        sw = score * weight
        swS = scoreS * weightS
        swT = scoreT * weightT
        if sw >= swS and sw >= swT:
            examReport[var]['w'] = weight
            examReport[var]['s'] = score
            examReport[var]['w·s'] = sw
        elif swS >= swT:
            examReport[var]['w'] = weightS
            examReport[var]['s'] = scoreS
            examReport[var]['w·s'] = swS
        else:
            examReport[var]['w'] = weightT
            examReport[var]['s'] = scoreT
            examReport[var]['w·s'] = swT

        finalscore += examReport[var]['w·s']

    totalscore = 10 * finalscore / len(solution_ex)
    examReport['Exam'] = {'Score': totalscore}

    return examReport
