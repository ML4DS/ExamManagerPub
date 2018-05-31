# This script contains the base class of all examProject classes, and some
# auxiliary methods.

import os
import copy
import shutil
import glob
import _pickle as pickle
from datetime import datetime
from babel.dates import format_date

import matplotlib.pyplot as plt

import csv
import pandas as pd
import numpy as np
from scipy.io import savemat


def addSlash(p):
    """
    If string p does not end with character '/', it is added.
    """
    if p.endswith('/'):
        return p
    else:
        return p + '/'


def removeSlash(p):
    """
    If string p does not end with character '/', it is added.
    """
    if p.endswith('/'):
        return p[:-1]
    else:
        return p


def getItems(x, z, n):
    """
    Takes n elements at random from list x without replacement (if possible).

    If the list x has less than n items, it is replicated as many times as
    necessary.

    All items in x that are in z must be selected.
    """

    # Input data sizes
    n_items = len(x)
    n_batch = np.ceil(float(n)/n_items).astype(int)

    # Shuffle elements of the original list, so as to randomize the item
    # selection.
    y = copy.copy(x)
    np.random.shuffle(y)

    # If n_items< n, we need to replicate the shuffled list up to get more than
    # n elements.
    y = np.tile(y, n_batch)
    y = y[0:n]

    # Re-shuffle to avoid periodicities
    np.random.shuffle(y)

    # If n>n_items, the mandatory  items in z have been necessarily selected.
    # Otherwise, we have forze selection.
    if n < n_items:

        # We have to insert the mandatory items, but maybe some of them are
        # are already in the list, so we proceed in three steps:
        # 1. Remove items of z not in x, because the cannot be inserted.
        #    WARNING: do not use z = list(set(z).intersection(x))
        #             because it affects to the random sequence generator
        z = [item for item in z if item in x]
        if len(z) > n:
            z = z[0:n]  # No more than n mandatory items can be included...
        # 2. Remove items of y that are in z (they will be added again later).
        #    WARNING: do not use y = list(set(y).difference(z))
        #             because it affects to the random sequence generator
        y = [item for item in y if item not in z]
        # 3. Remove some more items to free space for z.
        y = y[0:n-len(z)]
        # 4. Insert z
        y = y + z
        # 4. Reshuffle again to randomize the position of the mandatory tests.
        np.random.shuffle(y)

    return y


class ExamProject(object):
    """
    Exam is the general class encompassing all components of an evaluation
    project
    """

    def __init__(self, project_path):

        # This is the minimal information required to work with a project
        self.project_path = addSlash(project_path)
        self.state = self.project_path + 'state/'
        self.metadata = self.state + 'metadata.pkl'

    def create(self, f_struct, exam_class=None):
        """
        Creates a new exam project. To do so, it defines the main folder
        structure, and creates (or cleans) the project folder, specified in
        self.project_path
        """

        # Default project folders
        self.exam_class = exam_class
        self.f_struct = {
            'class_list': self.project_path + 'class_list/',
            'data4students': self.project_path + 'data4students/',
            'exam_statement': self.project_path + 'exam_statement/',
            'student_results': self.project_path + 'student_results/',
            'eval_results': self.project_path + 'eval_results/',
            'db_questions': self.project_path + 'dbFiltering/',
            'all_students': self.state + 'all_students.csv',
            'basename': 'Ex_',
            'student_notes_fname': 'student_notes.xlsx'}

        # Change default names by those specified in f_struct
        for s, v in f_struct.items():
            self.f_struct[s] = v

        # Add default subfolders
        if 'figq_path' not in self.f_struct:
            self.f_struct['figq_path'] = (self.f_struct['db_questions'] +
                                          'figs/')
        if 'template_path' not in self.f_struct:
            self.f_struct['template_path'] = (self.f_struct['db_questions'] +
                                              'LaTexTemplates/')
        if 'template_fpath' not in self.f_struct:
            self.f_struct['template_fpath'] = (self.f_struct['template_path'] +
                                               'Template.tex')

        # Check and clean project folder location
        if os.path.exists(self.project_path):
            print('Folder {} already exists.'.format(self.project_path))

            # Remove current backup folder, if it exists
            old_project_path = removeSlash(self.project_path) + '_old/'
            if os.path.exists(old_project_path):
                shutil.rmtree(old_project_path)

            # Copy current project folder to the backup folder.
            shutil.move(self.project_path, old_project_path)
            print('Moved to ' + old_project_path)

        # Create project folder and subfolders
        os.makedirs(self.project_path)

        # Create class_list subdirectory if it does not exist.
        # The remaininng subfolders will be created by the method that fills
        # them with content.
        if not os.path.exists(self.f_struct['class_list']):
            os.makedirs(self.f_struct['class_list'])

        # Save metadata
        metadata = {'f_struct': self.f_struct,
                    'exam_class': exam_class}
        if not os.path.exists(self.state):
            os.makedirs(self.state)
        with open(self.metadata, 'wb') as f:
            pickle.dump(metadata, f)

    def load(self, f_struct={}):
        """
        Loads an existing project, by reading the metadata file in the project
        folder.
        It can be used to modify file or folder names, or paths, by specifying
        the new names/paths in the f_struct dictionary.
        """

        # Check and clean project folder location
        if not os.path.exists(self.project_path):
            print('Folder {} does not exist.'.format(self.project_path))
            exit('You must create the project first')

        else:

            # Save project structure.
            with open(self.metadata, 'rb') as f:
                metadata = pickle.load(f)

            # Read metadata dictionary
            self.exam_class = metadata['exam_class']
            self.f_struct = metadata['f_struct']

            # Change default names by those specified in f_struct
            for s, v in f_struct.items():
                self.f_struct[s] = v
            metadata['f_struct'] = self.f_struct

            # Save project structure.
            with open(self.metadata, 'wb') as f:
                pickle.dump(metadata, f)

    def loadStudents(self):
        '''
        Generates a dataframe of students.
        For each student, name, nia and group are stored.
        Student data are taken from the xls files in the 'class_list' folder
        '''

        # #######################
        # Configurable parameters
        # #######################

        # Input folder containing the lists of students in the course.
        input_folder = self.f_struct['class_list']
        output_fname = self.f_struct['all_students']

        # ###############
        # Read input data
        # ###############

        # Read NIAs
        print('\n... Reading xls files')
        f_names = [f for f in os.listdir(input_folder)
                   if len(f) > 5 and f[-5:] == '.xlsx']

        # Set the group name by removing file extension.
        groups = [f.replace('.xlsx', '') for f in f_names]
        # Now we remove termination ' listadeclase'. This is specific of the
        # class_list files at UC3M.
        groups = [f.replace(' listadeclase', '') for f in groups]

        # Warn if there are no xls files in the class list folder
        if len(groups) == 0:
            print('WARNING: There are no groups in the class_list folder.')
            print('Download the class lists from Aula Global and place ' +
                  ' them in {0}.'.format(self.f_struct['class_list']))

            df_students = False

        else:

            sgroup = {}
            for f, g in zip(f_names, groups):
                # Read class list of group g
                print('      ' + f)
                sgroup[g] = pd.read_excel(input_folder + f)
                sgroup[g].rename(
                    columns={'Dirección de correo': 'Email address',
                             'Apellido(s)': 'Surname', 'Nombre': 'First name',
                             'NIU': 'NIA'},
                    inplace=True)
                # Add a new column identifying the group of each student
                sgroup[g]['group'] = g

            # Join all groups of students in a single dataframe.
            df_students = pd.concat(sgroup, axis=0, ignore_index=True)
            df_students['Full name'] = (df_students['First name'] + ' ' +
                                        df_students['Surname'])

            # Save dataframe
            print(df_students.head())
            df_students.to_csv(output_fname)

        return df_students

    def _makeExams(self, n_exams, exam_struct, q_ignore=[], q_mandatory=[],
                   seed=None):
        """
        Compose a set of exams.

        :Args:
            :n_exams:  Number of exames to be generated.
            :exam_struct: Structure of the requested exam.
                It is a dictionary defining the structure of each exam part
                    exam_struct = {key1: v1, key2: v2, ...}
                where
                    key1, key2,...
                are the names of each part. These names are not arbitrary: they
                must be equal to the tags used in the latex template to specify
                the location of the questions for each part. Also,
                    v1, v2, ...
                are lists defining the structure of each part. For instance,
                    s1 = [tag1, [tag2, tag3], [tag4, tag5]]
                defines part 1 with 3 questions:
                    - 1st question is about the topic specified in tag1
                    - 2nd question is about tag2 or tag3
                    - 3rd question is about tag4 or tag5.
                tag1, tag2, etc must be strings.
            :q_ignore: List of questions that must be ignored, and not used in
                this exam.
            :q_mandatory: List of questions that must appear in at least one of
                the exams.
            :seed:  Seed for the random exam generation. If None, no seed is
                used, so the set of exams generated by this script will change
                after each run.
        :Returns:
            :exam_pack: Dictionary containing the list of question for each of
                the generated exams.
        """

        ###################
        # ## Config Section
        ###################

        # Location and names of the source files
        dbq_path = self.f_struct['db_questions']
        figq_folder = self.f_struct['figq_path']
        template_path = self.f_struct['template_path']
        template_fpath = self.f_struct['template_fpath']

        # Location and names of the output files
        out_folder = self.f_struct['exam_statement']
        basename = self.f_struct['basename']

        # Required packages.
        # Write here the filename of all packages required to run the tex file.
        # The files should be available in the running folder.
        packages = ['mcode.sty']

        ########################
        # Read and organize data
        ########################

        # Date for the exam header
        now = datetime.now()
        date_en = format_date(now, format='MMM, YYYY', locale='en')
        date_es = format_date(now, format='MMM, YYYY', locale='es')
        # date = datetime.now().strftime("%B, %G%")

        # The code below assumes that each filename has the form
        # xx_topic_xxx.tex

        # Read tex filenames
        f_names = [f for f in os.listdir(dbq_path) if f.endswith('.tex')]

        # List of topics.
        # We assume here that the topics are embedded in the filename.
        topics = set([f.split('_')[1] for f in f_names])

        # Create dictionary tags-->questions:
        q = {}
        for t in topics:
            q[t] = [f for f in f_names if t in f]

        ##############################
        # Candidate list for each test
        ##############################

        # Create the list of candidates for each section and each question
        candidates = {}
        for sec in exam_struct:
            candidates[sec] = []
            for topic_subset in exam_struct[sec]:
                candidates[sec].append(
                    [el for t in topic_subset for el in q[t]
                     if el not in q_ignore])

        ######################
        # Create ouput folders
        ######################

        # Remove current output folder, if it exists
        if os.path.exists(out_folder):
            shutil.rmtree(out_folder)
        # Create new clean output folder and subfolders
        os.makedirs(out_folder)

        # Create exam subfolders
        self.exam_tags = []
        for k in range(n_exams):
            self.exam_tags.append(basename + str(k))
            out_folder_k = out_folder + basename + str(k)
            os.makedirs(out_folder_k)

            for f in packages:
                shutil.copy(template_path + f, out_folder_k + '/' + f)

        ############################
        # Compose and save all exams
        ############################

        # Read template file
        text = open(template_fpath).read()

        # Select questions for each exam
        # Note that the i-th item in q does NOT contains the questions of the
        # i-th exam, but the i-th questions of ALL exams

        # Reset random number generator
        if seed is not None:
            np.random.seed(seed)

        # Compute the matrix of selected questions.
        # SelectedQ[n][e] will be que n-th question of exam e
        selectedQ = {}
        for sec in exam_struct:
            selectedQ[sec] = []
            for options in candidates[sec]:
                selectedQ[sec].append(getItems(options, q_mandatory, n_exams))

        # This is the output variable containing the schema of each exam.
        exam_out = []

        # Read selected tests and problems from file
        for e in range(n_exams):

            # Print exam.
            print('Exam {0}: questions:'.format(e))

            # Start exam composition:
            exam_e = text.replace('<DATE_EN>', date_en)
            exam_e = exam_e.replace('<DATE_ES>', date_es)
            exam_e = exam_e.replace('figs/', figq_folder)

            qe = {}

            for sec in exam_struct:

                # List of questions for exam e
                qe[sec] = [selectedQ[sec][n][e]
                           for n in range(len(selectedQ[sec]))]

                # Print selected questions from this section
                print('    - {}'.format(qe[sec]))

                exam_quests = ''
                for q in qe[sec]:
                    # Read tests for exam e, from database files
                    with open(dbq_path + q, 'r') as f:
                        body = f.read()
                    exam_quests += '\\question[8] \n\n' + body + '\n\n'

                # Compose exam k
                exam_e = exam_e.replace('<' + sec + '>', exam_quests)

            # Save exam e to .tex file.
            fpath_exam_e = out_folder + basename + str(e) + \
                '/' + basename + str(e) + '.tex'
            with open(fpath_exam_e, "a+") as f:
                f.write(exam_e)

            # Save the exam list into a csv file
            qe_names = [q.split('.tex')[0] for q in qe]
            fpath_qe = out_folder + basename + str(e) + '/' + basename + \
                str(e) + '.csv'
            with open(fpath_qe, 'w') as f:
                wr = csv.writer(f, dialect='excel')
                wr.writerow(qe_names)

            exam_out.append(qe)

        return exam_out

    def prepareStResultsFolder(self):

        # Prepare student results folder, one per exam:
        if not os.path.exists(self.f_struct['student_results']):
            os.makedirs(self.f_struct['student_results'])

        for d in self.exam_tags:
            p = self.f_struct['student_results'] + d
            if not os.path.exists(p):
                os.makedirs(p)

    def generateDataFiles(self, generateData):
        '''
        Generates all data files for students in the xls files in the
        'class_list' folder, using the generateData function provided as an
        argument.
        '''

        # ###############
        # Read input data
        # ###############

        # Read NIAs
        print('\n... Reading student NIAs')
        df_students = pd.read_csv(self.f_struct['all_students'])

        # Identify groups
        groups = set(df_students['group'].tolist())

        # ######################
        # Prepare output folders
        # ######################

        # # Folder that will contain the output files.
        out_folder = self.f_struct['data4students']

        print('... Setting output folders')
        # Remove current output folder, if it exists
        if os.path.exists(out_folder):
            shutil.rmtree(out_folder)
        # Create new clean output folder and subfolders
        os.makedirs(out_folder)
        for g in groups:
            os.makedirs(out_folder + g)

        # ###############
        # Data generation
        # ###############

        # Generate and save
        print('... Computing data and saving output files')
        for k, st in df_students.iterrows():
            g = st['group']
            fpath = out_folder + g + '/' + str(st['NIA']) + '.mat'
            data = generateData(st['NIA'])
            savemat(fpath, data)

        print('... Done.\n')

    def getVarData(self, df, var):

        x = df[(var, 'w·s')].as_matrix()
        x = x[~np.isnan(x)]

        return x

    def getDataStats(self, x):

        # Compute mean and standard deviation
        m = np.mean(x)
        std = np.std(x)

        # Compute upper and lower standard deviations
        # Note that samples with x=m enter both xpos and xneg. This is likely
        # not correct. But I guess samples with x=m are few, if any.
        x_up = x[x >= m] - m
        x_down = m - x[x <= m]
        std_up = np.sqrt(np.mean(x_up**2))
        std_down = np.sqrt(np.mean(x_down**2))

        # Save stats
        stats = {'mean': m, 'std_up': std_up, 'std_down': std_down, 'std': std}

        return stats

    def analyzeExams(self):

        # ############
        # Loading data
        if 'student_notes_fname' not in self.f_struct:
            self.f_struct['student_notes_fname'] = 'student_notes.xlsx'

        # Read matlab files in the input directory tree
        datafiles = glob.glob(self.f_struct['eval_results'] + '**/' +
                              self.f_struct['student_notes_fname'],
                              recursive=True)

        # Read files
        df = pd.DataFrame()

        print('... concatenate all evaluation events')
        for dtfile in sorted(datafiles):

            df2 = pd.read_excel(dtfile, header=[0, 1])

            # Add to dataframe
            df = pd.concat([df, df2], axis=0)
            # df.sort_index(axis=1, inplace=True)

        # Read variable names
        allCols = df.columns.levels[0].tolist()
        metaCols = ['Delivery', 'Exam', 'Final', 'Unnamed: 0_level_0']
        varCols = list(set(allCols) - set(metaCols))

        # Get dictionary of variable: data.
        scores = {}
        stats = {}

        for var in varCols:

            scores[var] = self.getVarData(df, var)
            stats[var] = self.getDataStats(scores[var])

        N = len(varCols)
        menMeans = np.array([stats[var]['mean'] for var in varCols])
        # menStd = [stats[var]['std'] for var in varCols]
        std_up = [stats[var]['std_up'] for var in varCols]
        std_down = [stats[var]['std_down'] for var in varCols]
        err = np.array([std_down, std_up])

        ids_sorted = np.argsort(menMeans)
        mSorted = menMeans[ids_sorted]
        errSorted = err[:, ids_sorted]
        varSorted = [varCols[i] for i in ids_sorted]

        plt.figure()
        y_pos = np.arange(N)    # the x locations for the groups
        plt.barh(y_pos, mSorted, align='center', alpha=0.4, xerr=errSorted)
        plt.yticks(y_pos, varSorted)
        plt.xlabel('Average score')
        plt.title('Mean and one-sided standard deviations of scores')
        plt.show()

        # fig, ax = plt.subplots()
        # ind = np.arange(N)    # the x locations for the groups
        # width = 0.70         # the width of the bars
        # ax.barh(ind, menMeans, width, color='b', xerr=menStd)
        # ax.set_title('Average scores for each variable')
        # # ax.set_xticks(ind + width / 2)
        # ax.set_yticklabels(varCols)
        # # ax.tick_params(axis='x')
        # ax.autoscale_view()

        return

