# examManager
Code to generate statements and dataset for a lab exam, and to evaluate the results automatically

The examManager is a complete tool to generate and evaluate laboratory exams. It can be potentially applied to evaluation project from different courses, provided that the approriate database of questions and question solvers is provided.

This project contains several folders:

  * code: The python code of the examManager.
  * LabB12: A database containing question statements for blocks B1 and B2 from TMDE-UC3M course.
  * LabB3: A database containing question statements for blocks B3 from TMDE-UC3M course.
  * LabTD: A (micro) database containing question statements about regression and classification for a TD course.

## Managing existing exam databases.

Using this code to manage exams for `LabB12` and `LabB3`  and `TD`is easy, simply go to folder `code` and run

  python mainExamManager.py
  
and follow the instructions to create a new project, generate the exam statements, load the student lists, generate the datasets, load the student responses and evaluate the exams.

## Creating a new database

In order to apply this code to a new course (say, LabB4), you will need to:

  1. Generate a database of question statements, in a new folder LabB4, following a similar folder structure to that used in LabB12 and LabB3.
  2. Write an exam template, using tag `<QUESTIONS>` to indicate the location in the tex file of the questions selected for a specific exam.
  3. Write a module dbsolverB4.py, containing the solvers for each specific question in the .tex database.
  4. Specify the class `examProjectB4` in a file examProjectB4.py (inherited from the `examProject` class defined in `examProject.py`), in a similar way to examProjectB12.py and examProjectB3.py. Basically, this code should contain the rules for the exam generation in a function `makeExams(...)`




