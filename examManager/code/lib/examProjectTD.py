# This script contains a set of functions solving all questions stated in the
# filtering exam database

import numpy as np

# Local imports
import lib.examProject as ex


def generateData(nia):

    from sklearn.model_selection import train_test_split

    # #######################
    # Configurable parameters
    # #######################

    n_hash_max = 1000  # Maximum output value of the hash function

    # Dataset sizes
    N = 1200
    # NTest = 100

    Dim = 5

    # #########################
    # 1. Reset random generator
    # #########################

    # Simple hash taking values between 0 and nmax
    n_hash = nia % n_hash_max
    # Reset random number generator
    np.random.seed(n_hash)

    # #######################################
    # 2. Generate x, z, and partition numbers
    # #######################################

    # Generate mean and std for X
    xmean = 2 * np.random.randn(Dim,)
    xstd = np.abs(np.random.randn(Dim,))

    xtrC = xmean + xstd * np.random.randn(N, Dim)
    # xtstC = xmean + xstd * np.random.randn(NTest, Dim)

    # #######################################
    # 3. Generate labels
    # #######################################

    # Define the logistic function
    def logistic(t):
        return 1.0 / (1 + np.exp(-t))

    ztrC = np.hstack((np.ones((N, 1)), xtrC, 1e-4*(xtrC**2)))
    we = np.random.randn(2*Dim + 1)

    # Center the weights so that classes become quite balanced.
    we[0] = - np.dot(np.mean(ztrC[:, 1:], axis=0), we[1:])
    P1 = logistic(ztrC.dot(we))

    ytr = (P1 > np.random.rand(len(P1))).astype(int)

    # #######################################
    # 4. Split data in Train and VAl
    # #######################################

    xtrC, xvalC, ytrC, yvalC = train_test_split(xtrC, ytr, test_size=0.8)

    state = {}
    state['NIA'] = nia
    state['xtrC'] = xtrC
    state['xvalC'] = xvalC
    state['ytrC'] = ytrC
    state['yvalC'] = yvalC
    # state['xtstC'] = xtstC

    # ##########################
    # 5. Data for classification
    # ##########################

    x_min = 0.5
    x_max = 10
    ndata = 500
    sigma_eps = np.sqrt(0.4)

    mean_w = [0, 0, 0, 0]
    cov_w = [[1, 0.5, 0.5, 0.5],
             [0.5, 1, 0.5, 0.5],
             [0.5, 0.5, 1, 0.5],
             [0.5, 0.5, 0.5, 1]]

    w_true = np.random.multivariate_normal(mean_w, cov_w).T

    # Training datapoints
    xtrR = np.linspace(x_min, x_max, ndata)
    strR = (w_true[0] + w_true[1] * xtrR + w_true[2] * np.log(np.abs(xtrR)) +
            w_true[3] * np.multiply(xtrR, np.log(np.abs(xtrR))) +
            sigma_eps * np.random.randn(ndata))

    data = {'xtrC': xtrC, 'xvalC': xvalC, 'ytrC': ytrC, 'yvalC': yvalC,
            'xtrR': xtrR, 'strR': strR, 'state': state}
    # 'xtstC': xtstC, 'xtrR': xtrR, 'strR': strR, 'state': state}

    return data


def generateDataTD_deprecated(nia):

    # #######################
    # Configurable parameters
    # #######################

    n_hash_max = 1000  # Maximum output value of the hash function

    # Dataset sizes
    N = 1200
    # NTest = 100

    Dim = 5

    # #########################
    # 1. Reset random generator
    # #########################

    # Simple hash taking values between 0 and nmax
    n_hash = nia % n_hash_max
    # Reset random number generator
    np.random.seed(n_hash)

    # #######################################
    # 2. Generate x, z, and partition numbers
    # #######################################

    # Generate mean and std for X
    xmean = 2 * np.random.randn(Dim,)
    xstd = np.abs(np.random.randn(Dim,))

    xtrC = xmean + xstd * np.random.randn(N, Dim)
    # xtstC = xmean + xstd * np.random.randn(NTest, Dim)

    # #######################################
    # 3. Generate labels
    # #######################################

    # Define the logistic function
    def logistic(t):
        return 1.0 / (1 + np.exp(-t))

    ztrC = np.hstack((np.ones((N, 1)), xtrC, 1e-4*(xtrC**2)))
    we = np.random.randn(2*Dim+1)

    P1 = logistic(ztrC.dot(we))

    if np.sum(P1 > 0.5) == 0 or np.sum(P1 <= 0.5) == 0:
        th = np.mean(P1)
        print(th)
    else:
        th = 0.5

    ytr = []

    for el in P1:
        if el > th:
            ytr.append(1)
        else:
            ytr.append(0)

    ytr = np.array(ytr)

    if np.sum(ytr) == 0 or np.sum(ytr == 0) == 0:

        th = np.median(P1)
        ytr = []
        for el in P1:
            if el > th:
                if np.random.rand() > 0.01:
                    ytr.append(1)
                else:
                    ytr.append(0)
            else:
                if np.random.rand() > 0.01:
                    ytr.append(0)
                else:
                    ytr.append(1)

        ytr = np.array(ytr)

    # #######################################
    # 4. Split data in Train and VAl
    # #######################################

    from sklearn.model_selection import train_test_split

    xtrC, xvalC, ytrC, yvalC = train_test_split(xtrC, ytr, test_size=0.8)

    state = {}
    state['NIA'] = nia
    state['xtrC'] = xtrC
    state['xvalC'] = xvalC
    state['ytrC'] = ytrC
    state['yvalC'] = yvalC
    # state['xtstC'] = xtstC

    # ##########################
    # 5. Data for classification
    # ##########################

    x_min = 0.5
    x_max = 10
    ndata = 500
    sigma_eps = np.sqrt(0.4)

    mean_w = [0, 0, 0, 0]
    cov_w = [[1, 0.5, 0.5, 0.5],
             [0.5, 1, 0.5, 0.5],
             [0.5, 0.5, 1, 0.5],
             [0.5, 0.5, 0.5, 1]]

    w_true = np.random.multivariate_normal(mean_w, cov_w).T

    # Training datapoints
    xtrR = np.linspace(x_min, x_max, ndata)
    strR = (w_true[0] + w_true[1] * xtrR + w_true[2] * np.log(np.abs(xtrR)) +
            w_true[3] * np.multiply(xtrR, np.log(np.abs(xtrR))) +
            sigma_eps * np.random.randn(ndata))

    data = {'xtrC': xtrC, 'xvalC': xvalC, 'ytrC': ytrC, 'yvalC': yvalC,
            'xtrR': xtrR, 'strR': strR, 'state': state}
    # 'xtstC': xtstC, 'xtrR': xtrR, 'strR': strR, 'state': state}

    return data


class ExamProjectTD(ex.ExamProject):

    def makeExams(self, n_exams, exam_struct=None, q_ignore=[], q_mandatory=[],
                  seed=None):
        """
        Compose a set of exams.
        This script only sets default values for the input variables and
        calls to __makeExams in the base class, which makes all of the work.

        WARNING: THIS VERSION IS CURRENTLY UNDER CONSTRUCTION, BECAUSE NO
                 DATABASE OF TD EXAMS IS CURRENTLY AVAILABLE

        :Args:
            :n_exams:  Number of exames to be generated.
            :exam_struct: Structure of the requested exam. If None, a default
                structure is used.
                A exam scturture is a list of lists containing any of the
                following elements:
                    'estimate', 'model', 'predict', 'evaluate', 'lms', 'rls',
                    'estimate2'
                Not all exam structures are allowed. Thus, the use of this
                variable is advised only for informed users...
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
            exam_struct = {'QUESTIONS': []}

        exam_out = super(ExamProjectTD, self)._makeExams(
            n_exams, exam_struct, q_ignore, q_mandatory, seed=seed)

        return exam_out

    def generateDataFiles(self):

        # Call the data generator in the parent class, passing as argument the
        # specific generateData method from the child class.
        super(ExamProjectTD, self).generateDataFiles(generateData)
