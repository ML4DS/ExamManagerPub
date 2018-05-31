# This script contains a set of functions solving all questions stated in the
# filtering exam database

import numpy as np
from scipy.linalg import toeplitz

# Local imports
import lib.examProject as ex


def generateData(nia):

    # #######################
    # Configurable parameters
    # #######################

    n_hash_max = 1000  # Maximum output value of the hash function

    # Filter size
    M_min = 10
    M_max = 30

    # Dataset sizes
    N = 400
    Nu = 2*N
    NuTest = M_max + int(N / 10)

    # Noise variance bounds
    varN_min = 0.02   # Minimum noise variance
    varN_max = 0.04   # Minimum noise variance

    # #########################
    # 1. Reset random generator
    # #########################

    # Simple hash taking values between 0 and nmax
    n_hash = nia % n_hash_max
    # Reset random number generator
    np.random.seed(n_hash)

    # ##################
    # 2. Generate filter
    # ##################

    # The filter is the cascade composition of two filters, h and g

    # Filter h. The values of this filter are given by a problem statement
    h = np.array([1, 0.2])

    # Filter g = [1, 0.4, 0.2, g3, g4, ...]
    # 1.- Generate the random part of the filter as a uniform between 20 and 40
    Ng = M_min + int(np.random.rand(1) * (M_max - M_min)) - len(h) + 1
    varS = .5 / Ng
    g = varS * np.random.randn(Ng)
    g[0:4] = [2, 0.4, 0.2, 0.1]

    # Filter
    s = np.convolve(h, g)
    Ns = len(s)

    # print('s = {0}'.format(s))
    # print('Ns = {0}'.format(Ns))

    # alfa = float(n_hash) / 1000 + 0.8
    # spikes = np.linspace(0, M-1, 5, dtype=int)
    # for i in range(5):
    #     s[spikes[i]] = alfa**(i - 1)

    # %%%%%%%%%%%%%%%%%%%%%%
    # % 4.- Generate x
    # %%%%%%%%%%%%%%%%%%%%%%

    # Generate the noise variances
    varN = varN_min + (varN_max - varN_min) * float(n_hash) / n_hash_max

    u = np.random.randn(Nu,)
    uTest = np.random.randn(NuTest,)

    col = np.zeros(Ns)
    col[0] = u[0]
    U0 = toeplitz(col, u)

    x = U0.T.dot(s) + np.sqrt(varN) * np.random.randn(Nu)
    x = x[0:N]

    state = {}
    state['NIA'] = nia
    state['x'] = x
    state['u'] = u
    state['uTest'] = u
    state['varS'] = varS
    state['varN'] = varN
    state['M'] = Ns

    data = {'x': x, 'u': u, 'uTest': uTest, 'varS': varS, 'varN': varN,
            'M': Ns, 'state': state}

    return data


class ExamProjectB3(ex.ExamProject):

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
                    'estimate', 'model', 'predict', 'evaluate', 'lms', 'rls',
                    'estimate2'

                Not all exam structures are allowed. Defining
                    E = 'estimate'
                    S = 'estimate2'
                    M = 'model'
                    P = 'predict'
                    V = 'evaluate'
                    L = 'lms'
                    R = 'rls'
                any exam should satisfy the following regular expression:
                    [ [E|S]+ [E|S|M|P|V|L|R]* ]+
                Basically, this means that E or S should appear at least once
                at the beginning.
                Any exam without this structure may have an inconsisten
                statement and produce errors in the exam solver.
                A recommended exam statement should be a subsequence of
                    E-M-P-V-S-L-R
                (E and S refer both to estimation questions, but S are more
                difficult). To reduce the number of questions in the exam,
                any element in the chain can be ommited unless E.

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
            exam_struct = {'QUESTIONS': [['estimate'],
                                         ['model'],
                                         ['predict'],
                                         ['evaluate', 'lms', 'rls'],
                                         ['estimate2']]}

        exam_out = super(ExamProjectB3, self)._makeExams(
            n_exams, exam_struct, q_ignore, q_mandatory, seed=seed)

        return exam_out

    def generateDataFiles(self):

        # Call the data generator in the parent class, passing as argument the
        # specific generateData method from the child class.
        super(ExamProjectB3, self).generateDataFiles(generateData)
