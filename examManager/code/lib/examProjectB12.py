# This script contains a set of functions solving all questions stated in the
# filtering exam database

import numpy as np

# Local imports
import lib.examProject as ex


def generateData(nia):

    # #######################
    # Configurable parameters
    # #######################

    n_hash_max = 500  # Maximum output value of the hash function

    # #########################
    # 1. Reset random generator
    # #########################

    # Simple hash taking values between 0 and nmax
    n_hash = nia % n_hash_max
    # Reset random number generator
    np.random.seed(n_hash)

    # ######################
    # 2. Data for Regression
    # ######################

    ntr = 300          # No. of training samples
    ntst = 100          # No. of test samples
    D = 5              # Input data dimension
    xTrain = np.random.randn(ntr, D)
    xVal = np.random.randn(ntst, D)
    # Xe = np.concatenate((np.ones((ntr, 1)), xTrain), axis=1)

    r = n_hash / n_hash_max
    w = np.array([0.1, r, 1-r, r/2, (r+1)/3, r/3, -r/2, -r/10, (1-r)/5])
    w[w == 0] = 0.01
    sTrain = (w[0] + np.dot(xTrain, w[1:6]) +
              w[6] * xTrain[:, 1] * xTrain[:, 2] +
              w[7] * xTrain[:, 3]**2 + w[8] * np.log(xTrain[:, 4]**2))
    sVal = (w[0] + np.dot(xVal, w[1:6]) +
            w[6] * xVal[:, 1] * xVal[:, 2] +
            w[7] * xVal[:, 3]**2 + w[8] * np.log(xVal[:, 4]**2))

    # We provide a linear transformation of the data
    v = 2 * np.random.randn(1, 5)**2 + 3
    m = 2 * np.random.randn(1, 5)

    xTrain = xTrain * np.sqrt(v) + m
    xVal = xVal * np.sqrt(v) + m

    flag = np.random.rand(ntr, D) > 0.9
    xTrainLost = -999 * flag + xTrain * (1 - flag)

    # ########################
    # 3. Data for Clasfication
    # ########################

    D = 7
    ntr = 200
    ntst = 50

    v = 2       # Variance can range between 0.5 and 1.5
    m0 = np.random.rand(1)
    m1 = m0 + 1
    P0 = 0.35 + .3 * np.random.rand(1)

    Xtrain = np.zeros((0, D))
    ytrain = np.zeros((0, 1))
    Xtest = np.zeros((0, D))

    for k in range(ntr):
        if np.random.rand() > P0:
            ytrain = np.concatenate((ytrain, [[1]]))
            Xtrain = np.concatenate(
                (Xtrain, np.random.randn(1, D) * np.sqrt(v) + m1))
        else:
            ytrain = np.concatenate((ytrain, [[0]]))
            Xtrain = np.concatenate(
                (Xtrain, np.random.randn(1, D) * np.sqrt(v) + m0))

    for k in range(ntst):
        if np.random.rand() > P0:
            Xtest = np.concatenate(
                (Xtest, np.random.randn(1, D) * np.sqrt(v) + m1))
        else:
            Xtest = np.concatenate(
                (Xtest, np.random.randn(1, D) * np.sqrt(v) + m0))

    state = {}
    state['NIA'] = nia
    # state['x'] = x

    data = {'xTrain': xTrain, 'xVal': xVal, 'sTrain': sTrain, 'sVal': sVal,
            'xTrainLost': xTrainLost,
            'Xtrain': Xtrain, 'Xtest': Xtest, 'ytrain': ytrain,
            'state': state}

    return data


class ExamProjectB12(ex.ExamProject):

    def makeExams(self, n_exams, exam_struct=None, q_ignore=[], q_mandatory=[],
                  seed=None):
        """
        Compose a set of exams.
        This script only sets default values for the input variables and
        calls to __makeExams in the base class, which makes all of the work.

        :Args:
            :n_exams:  Number of exames to be generated.
            :exam_struct: Structure of the requested exam. If None, a default
                structure is used.
                A exam structure is a list of lists containing any of the
                following elements:
                    'preproc', 'calculow', 'estimacion', 'evalua',
                    'analisis', 'clasx1', 'evalx1', 'sufstat', 'clasxD',
                    'evalxD'

                Not all exam structures are allowed. Defining
                    P = 'preproc'
                    W = 'calculow'
                    E = 'estimacion'
                    V ='evalua'
                    A = 'analisis'
                    C = 'clasx1'
                    L = 'evalx1'
                    S = 'sufstat'
                    D = 'clasxD',
                    X = 'evalxD'
                any exam should satisfy the following regular expression:
                    [ P? W+ [E|V]+ ]+   (for the regresion part)
                    [ A? C+ L* S? [D X?]? ]+
                Any exam without this structure may have an inconsisten
                statement and produce errors in the exam solver.
                A recommended exam statement should satisfy
                    P-W-E-V-A-C-L-S-D-X
                To reduce lengh, any element in the chain can be ommited,
                unless W and C or D.

            :seed:  Seed for the random exam generation. If None, no seed is
                used.
        :Returns:
            :exam_pack: Dictionary containing the list of question for each
                of the generated exams.
        """

        # Structure of the exam.
        # Each item in the list contains the topics assigned to each question.
        # These topic names must coincide with those embedded in the file names
        if exam_struct is None:
            exam_struct = {'QUESTIONS1': [['preproc'],
                                          ['calculow'],
                                          ['estimacion', 'evalua']],
                           'QUESTIONS2': [['analisis'],
                                          ['clasx1'],
                                          ['clasxD']]}

        exam_out = super(ExamProjectB12, self)._makeExams(
            n_exams, exam_struct, q_ignore, q_mandatory, seed=seed)

        return exam_out

    def generateDataFiles(self):

        # Call the data generator in the parent class, passing as argument the
        # specific generateData method from the child class.
        super(ExamProjectB12, self).generateDataFiles(generateData)
