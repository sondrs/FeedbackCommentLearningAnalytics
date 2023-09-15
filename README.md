# Learning analytics: Mining student comments to increase course evaluation insights

This is the repository with all the programming for the Course Evaluation Analysis App created in the design science research process undertaken as part of the master thesis on text mining of course evaluation feedback comments, titled "Learning analytics: Mining student comments to increase course evaluation insights"

The repo contains all code but not all resources used. 

First and foremost, the course evaluation data, which should be in .csv, and put in the "Input" folder is not included as the data is owned by Kristiania. The filename used as default value in the functions in the implementation is "Evaluation_2020_2021.csv", and the file have questions/metrics spread out in comlumns and responses/samples in rows.

Secondly, training resources used such as the NoReC and ToN corpora which belong in folders 'norec' and 'talk-of-norway' both within the 'resources' folder, are not included. 

Note that the repo utilise Git LFS (Large File Storage) which must be set up in case of any pulls:
Ensure LFS installed, see https://git-lfs.github.com/

Then in destined folder, run:

git clone https://github.com/sondrs/FeedbackCommentLearningAnalytics.git

git lfs pull

The implementation utilise a variety of python libraries. The most important ones are Plotly Dash, JupyterDash (because of the use of Jupyter for the start page), Numpy, pandas, SciPy, scikit-learn, joblib, but there are likely, depending on what is already on any given system installation of python, other dependencies as well that require installation through pip, conda or similar. Packaging the whole app including the python code and libraries used in one executable was outside the scope of the development undertaken in this thesis.
