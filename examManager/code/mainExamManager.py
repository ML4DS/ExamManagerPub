# -*- coding: utf-8 -*-
"""
Main program for the exam management application. It allows to create an
evaluation project, generate exams and datasets and get everything ready for
the evaluation (provided that the user locates the appropriate student data
files in the required folders)

Created on Jan 02 2018

@author: Jesús Cid Sueiro
         (query_int and clear methods taken from code by Jeronimo Arenas)

Version: 1.0 (Jan, 2018). Integrates management of exam classes B12 and B3.
"""

from __future__ import print_function    # For python 2 copmatibility
import os
import platform
import argparse

# Import all available evaluator classes (only the one selected by the user
# will be used, though)
import lib.examProject as exBase
import lib.dbEvaluator as dbEval

import lib.examProjectB12 as exB12
import lib.examProjectB3 as exB3
import lib.examProjectTD as exTD

from lib.dbSolverB12 import solveQuestion as solveQ_B12
from lib.dbSolverB3 import solveQuestion as solveQ_B3
from lib.dbSolverTD import solveQuestion as solveQ_TD


def query_int(pregunta, opciones):

    """
    Query the user for a number in predefined interval
        :param pregunta: Question that will be shown to the user
        :param opciones: Available choices
    """

    opcion = None
    while opcion not in opciones:
        opcion = input(pregunta + ' [' + str(opciones[0]) + '-' +
                       str(opciones[-1]) + ']: ')
        try:
            opcion = int(opcion)
        except:
            print('Introduzca un número')
            opcion = None
    return opcion


def clear():

    """Cleans terminal window
    """
    # Checks if the application is running on windows or other OS
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


# ################################
# Main body of application

clear()
print('*********************')
print('*** LAB EVALUATOR ***')

var_exit = False
time_delay = 3

# ########################
# Configurable parameters:
# ########################

# Number of exams
n_exams = 4

# Seed for the random exam generation
seed = 201712

# Location and names of the source files. This is specific of the exam class
f_structB12 = {'db_questions': '../LabB12/dbQuestions/',
               'figq_path': '../../../LabB12/dbQuestions/figs/',
               'template_path': '../LabB12/LaTexTemplates/',
               'template_fpath': '../LabB12/LaTexTemplates/TemplateB12Lab.tex',
               'basename': 'ExLabB12_'}

f_structB3 = {'db_questions': '../LabB3/dbFiltering/',
              'figq_path': '../../../LabB3/dbFiltering/figs/',
              'template_path': '../LabB3/LaTexTemplates/',
              'template_fpath': '../LabB3/LaTexTemplates/TemplateB3Lab.tex',
              'basename': 'ExLabB3_'}

f_structTD = {'db_questions': '../LabTD/dbQuestions/',
              'figq_path': '../../../LabTD/dbQuestions/figs/',
              'template_path': '../LabTD/LaTexTemplates/',
              'template_fpath': '../LabTD/LaTexTemplates/TemplateTDLab.tex',
              'basename': 'ExLabTD_'}

# Exam structures.
exam_struct = {'B12': {'QUESTIONS1': [['preproc'],
                                      ['calculow'],
                                      ['estimacion', 'evalua']],
                       'QUESTIONS2': [['analisis'],
                                      ['clasx1'],
                                      ['evalx1'],
                                      ['clasxD'],
                                      ['evalxD']]},
               'B3': {'QUESTIONS': [['estimate'],
                                    ['model'],
                                    ['predict'],
                                    ['evaluate', 'lms', 'rls'],
                                    ['estimate2']]},
               'TD': {}}

# Exam questions to be ignored
q_ignore = {'B12': [],
            'B3': [],
            'TD': []}

# Exam questions that must be included
q_mandatory = {'B12': [],
               'B3': [],
               'TD': []}


# ####################
# Read input arguments
# ####################

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default=None,
                    help="path to an existing evaluation project")
args = parser.parse_args()

# Read argumens
if args.p is None:
    isProjectActive = False
else:

    project_path = args.p

    # The following is actually a duplicate of code existing below under
    # optioni1 = 2. Not very nice, but good for a lazy coder.
    # This is a trick to load the exam class from the existing class
    exc = exBase.ExamProject(project_path)
    exc.load()
    exam_class = exc.exam_class

    # Now we load the exam project by creating an object from the appropriate
    # exam class
    print("Loading exam from class {}".format(exam_class))
    if exam_class == 'B12':
        examB = exB12.ExamProjectB12(project_path)
    elif exam_class == 'B3':
        examB = exB3.ExamProjectB3(project_path)
    elif exam_class == 'TD':
        examB = exTD.ExamProjectTD(project_path)
    examB.load()

    # This is to activate the expansion of the menu options
    isProjectActive = True


# ########################
# User interaction
# ########################

while not var_exit:

    print('\nAvailable options:')
    print(' 1. Create new evaluation project')
    print(' 2. Load existing project')
    if isProjectActive:
        print(' 3. Make exams')
        print(' 4. Load student groups')
        print(' 5. Generate data files for the students')
        print(' 6. Evaluate students')
        print(' 7. Analize evaluation results')
        n_opt = 8
    else:
        n_opt = 3
    print(' 0. Exit the application\n')

    option1 = query_int('What would you like to do?', range(n_opt))

    if option1 == 0:

        # Set a marker to exit the application
        var_exit = True

    elif option1 == 1:

        # ###############################
        # Create a new evaluation project

        project_path = input(
            'Write the desired full path to the new project: ')

        print('\nSelect the class of exam:')
        print(' 1. [B12]: TMDE lab exam about Regression and Decision')
        print(' 2. [B3]: TMDE lab exam about Linear Filtering')
        print(' 3. [TD]: TD lab exam about Regression and Classification')
        print(' 0. Exit the application\n')
        n_opt11 = 4
        option11 = query_int('What would you like to do?', range(n_opt11))

        if option11 == 0:
            var_exit = True
        else:
            # Select the appropriate tag for the exam class.
            exam_class = ['B12', 'B3', 'TD'][option11 - 1]

            # Select evaluation project
            if option11 == 1:
                f_struct = f_structB12
                examB = exB12.ExamProjectB12(project_path)
            elif option11 == 2:
                f_struct = f_structB3
                examB = exB3.ExamProjectB3(project_path)
            else:
                f_struct = f_structTD
                examB = exTD.ExamProjectTD(project_path)

            # Create evaluation project
            examB.create(f_struct, exam_class)

            # This is to activate the expansion of the menu options
            isProjectActive = True

            print("Project {0} created.".format(project_path))
            print("Project metadata saved in {0}".format(examB.metadata))
            print(("\nThe next step should be creating the exam statements. " +
                   "This script will generate {0} random exams using seed " +
                   "{1}. You might want to change these parameters at the " +
                   "begining of this script").format(n_exams, seed))

    elif option1 == 2:

        # ########################
        # Load an existing project

        project_path = input('Write the path to the project: ')

        # This is a trick to load the exam class from the existing class
        exc = exBase.ExamProject(project_path)
        exc.load()
        exam_class = exc.exam_class

        # Now we load the exam project by creating an object from the
        # appropriate exam class
        print("Loading exam from class {}".format(exam_class))
        if exam_class == 'B12':
            examB = exB12.ExamProjectB12(project_path)
        elif exam_class == 'B3':
            examB = exB3.ExamProjectB3(project_path)
        elif exam_class == 'TD':
            examB = exTD.ExamProjectTD(project_path)
        examB.load()

        # This is to activate the expansion of the menu options
        isProjectActive = True

        print("Project {0} loaded.".format(project_path))
        print(("\nThe next step should be creating the exam statements. This" +
               " script will generate {0} random exams using seed {1}. You " +
               "might want to change these parameters at the begining of " +
               "this script").format(n_exams, seed))

    elif option1 == 3:

        # #############################################
        # Generate random exams from the selected class

        # examB.makeExams(n_exams, seed)
        exam_scheme = examB.makeExams(
            n_exams, exam_struct[exam_class], q_ignore[exam_class],
            q_mandatory[exam_class], seed=seed)
        examB.prepareStResultsFolder()

        print('{0} exams created using seed {1}'.format(n_exams, seed))
        print('Source files saved in {0}'.format(
            examB.f_struct['exam_statement']))

        for i, x in enumerate(exam_scheme):

            print('Testing exam {} ...'.format(i))

            # Get exam questions
            quests = []
            for sec, qs in x.items():
                quests.extend(qs)
            # Remove file extensions
            quests = [q.split('.tex')[0] for q in quests]

            # Solve exam qust
            # Solving all questions for all possible dataset.
            n_max = 500
            for nia in range(n_max):

                print('-- User {0} out of {1}\r'.format(nia, n_max), end='')
                # flush=True)
                if exam_class == 'B12':
                    data = exB12.generateData(nia)
                    results, scoring = dbEval.solveExam(quests, data,
                                                        solveQ_B12)
                elif exam_class == 'B3':
                    data = exB3.generateData(nia)
                    results, scoring = dbEval.solveExam(quests, data,
                                                        solveQ_B3)
                else:
                    data = exTD.generateData(nia)
                    results, scoring = dbEval.solveExam(quests, data,
                                                        solveQ_TD)

        print('--- Systematic Test OK: no errors detected.')

        print(("\nNow you can load the student lists. To do so, download the" +
               " xlsx student lists from Aula Global and place them in " +
               "folder {0}.").format(examB.f_struct['class_list']))

    elif option1 == 4:

        # ###################
        # Load student groups

        # This step will work only if the xls student lists have been placed
        # in the appropriate folders
        df_students = examB.loadStudents()

        if df_students is not False:
            print('A csv containing the students from all groups has been ' +
                  'saved in {0}'.format(examB.f_struct['all_students']))
        print("\nNow you are ready to generate the data files for students")

    elif option1 == 5:

        # ########################################
        # Generate the data files for the students

        examB.generateDataFiles()

        print('Data files for students generated and saved in {0}'.format(
            examB.f_struct['data4students']))
        print(("\nThis completes the software execution before the exam. " +
               "After the exam, download the student responses and place " +
               "them in {0}, in the subfolder with the same name than the " +
               "folder containing the corresponding exam statement. After " +
               "That, open the evaluation notebook in your browser an" +
               "complete the evaluation").format(
               examB.f_struct['student_results']))

    elif option1 == 6:

        # #################
        # Evaluate students

        print("The student evaluation is currently not avialable from here." +
              " To evaluate the students run ")
        print("    jypyter notebook ExamEvaluator.ipynb")
        print("Select the exam to evaluate and execute notebook cells")

    elif option1 == 7:

        # #######################
        # Analyze student results

        exam_scheme = examB.analyzeExams()
