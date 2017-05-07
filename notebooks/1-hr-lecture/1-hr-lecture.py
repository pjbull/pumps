
# coding: utf-8

# ![DrivenData Logo](https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png)

# In[1]:

from __future__ import print_function

import os
import sys

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

PROJ_ROOT = os.path.join(os.pardir)

print(os.path.abspath(PROJ_ROOT))





# # The [watermark](https://github.com/rasbt/watermark) extension
# 
# Tell everyone when your notebook was run, and with which packages. This is especially useful for nbview, blog posts, and other media where you are not sharing the notebook as executable code.

# In[2]:

get_ipython().system('pip install watermark')


# In[3]:

# once it is installed, you'll just need this in future notebooks:
get_ipython().magic('load_ext watermark')


# In[4]:

get_ipython().magic('watermark -a "Peter Bull" -d -t -v -p numpy,pandas')


# In[5]:

get_ipython().magic('pinfo %watermark')


# ## Laying the foundation
# 
# Continuum's `conda` tool provides a way to create [isolated environments](http://conda.pydata.org/docs/using/envs.html). In fact, you've already seen this at work if you followed the [pydata setup](https://github.com/drivendata/pydata-setup) instructions to setup your machine for this tutorial. The `conda env` functionality let's you created an isolated environment on your machine for 
# 
#  - Start from "scratch" on each project
#  - Choose Python 2 or 3 as appropriate
# 
# To create an empty environment:
# 
#  - `conda create -n <name> python=3`
# 
# **Note: `python=2` will create a Python 2 environment; `python=3` will create a Python 3 environment.**
# 
# 
# To work in a particular virtual environment:
# 
#  - `source activate <name>`
#  
# To leave a virtual environment:
# 
#  - `source deactivate`
# 
# **Note: on Windows, the commands are just `activate` and `deactivate`, no need to type `source`.**
# 
# There are other Python tools for environment isolation, but none of them are perfect. If you're interested in the other options, [`virtualenv`](https://virtualenv.pypa.io/en/stable/) and [`pyenv`](https://github.com/yyuu/pyenv) both provide environment isolation. There are _sometimes_ compatibility issues between the Anaconda Python distribution and these packages, so if you've got Anaconda on your machine you can use `conda env` to create and manage environments.
# 
# -------------------
#  
# **`#lifehack`: create a new environment for every project you work on**
# 
# ------------
# 

# ## The `pip` [requirements.txt](https://pip.readthedocs.org/en/1.1/requirements.html) file
# 
# It's a convention in the Python ecosystem to track a project's dependencies in a file called `requirements.txt`. We recommend using this file to keep track of your MRE, "Minimum reproducible environment".
# 
# Conda
# 
# -----------
# 
# **`#lifehack`: never again run `pip install <package>`. Instead, update `requirements.txt` and run `pip install -r requirements.txt`**
# 
# -------
# 

# In[6]:

# what does requirements.txt look like?
print(open(os.path.join(PROJ_ROOT, 'requirements.txt')).read())


# The format for a line in the requirements file is:
# 
#  | Syntax | Result |
#  | --- | --- |
#  | `package_name` | for whatever the latest version on PyPI is |
#  | `package_name==X.X.X` | for an exact match of version X.X.X |
#  | `package_name>=X.X.X` | for at least version X.X.X |
#  
# Now, contributors can create a new virtual environment (using conda or any other tool) and install your dependencies just by running:
# 
#  - `pip install -r requirements.txt`

# --------------
# 
# 
# 
# 
# 
# 
# 
# 
# 
# -------------

# # Let's get to the data!
# 
# We've got our environment set up, we're tracking the packages that we use, and we've got a standard folder structure. Now that all of that is working we can 
# 

# In[7]:

## Try adding parameter index=0
pump_data_path = os.path.join(PROJ_ROOT,
                              "data",
                              "raw",
                              "pumps_train_values.csv")

df = pd.read_csv(pump_data_path, index_col=0)
df.head(3)


# In[8]:

get_ipython().magic('pinfo pd.read_csv')


# In[ ]:




# In[9]:

df.describe()


# In[10]:

## Paste for 'construction_year' and plot
## Paste for 'gps_height' and plot
plot_data = df['amount_tsh']
sns.kdeplot(plot_data, bw=50)
plt.show()

plot_data = df['construction_year']
sns.kdeplot(plot_data, bw=1)
plt.show()

plot_data = df['gps_height']
sns.kdeplot(plot_data, bw=0.2)
plt.show()


# In[11]:

def kde_plot(dataframe, variable, upper=None, lower=None, bw=0.1):
    """ Plots a density plot for a variable with optional upper and
        lower bounds on the data (inclusive).
    """
    plot_data = dataframe[variable]
    
    if upper is not None:
        plot_data = plot_data[plot_data <= upper]
    if lower is not None:
        plot_data = plot_data[plot_data >= lower]

    sns.kdeplot(plot_data, bw=bw)
    
    plt.savefig(os.path.join(PROJ_ROOT, 'reports', 'figures', '{}.png'.format(variable)))
    
    plt.show()


# In[12]:

kde_plot(df, 'amount_tsh', bw=100, lower=0)
kde_plot(df, 'construction_year', bw=1, lower=1000, upper=2016)
kde_plot(df, 'gps_height', bw=0.1)


# -------
# 
# 
# -------
# 
# 
# # Writing code for reproducibility
# 
# So, we've got some invalid data in this dataset. For example, water pumps installed before in the year 0. We'll want to have a function to load and clean this data since we will probably be using this data in multiple datasets.
# 
# Here's a first pass at a function that will do that for us. Now, we've got the function implemented in the notebook, but let's bring it to a standalone file.
# 
# We'll copy these functions into:
# `src/features/build_features.py`

# In[13]:

def awesome_function(s):
    from IPython.display import display, HTML
    css = """
        .blink {
            animation-duration: 1s;
            animation-name: blink;
            animation-iteration-count: infinite;
            animation-timing-function: steps(2, start);
        }
        @keyframes blink {
            80% {
                visibility: hidden;
            }
        }"""

    to_show = HTML(
        '<style>{}</style>'.format(css) +
        '<p class="blink"> {} IS AWESOME!!!!! </p>'.format(s)
    )
    display(to_show)


def remove_invalid_data(path):
    """ Takes a path to a water pumps csv, loads in pandas, removes
        invalid columns and returns the dataframe.
    """
    df = pd.read_csv(path, index_col=0)

    # preselected columns
    useful_columns = ['amount_tsh',
                      'gps_height',
                      'longitude',
                      'latitude',
                      'region',
                      'population',
                      'construction_year',
                      'extraction_type_class',
                      'status_group',
                      'management_group',
                      'quality_group',
                      'source_type',
                      'waterpoint_type']

    df = df[useful_columns]

    invalid_values = {
        'amount_tsh': {0: np.nan},
        'longitude': {0: np.nan},
        'installer': {0: np.nan},
        'construction_year': {0: np.nan},
    }

    # drop rows with invalid values
    df.replace(invalid_values, inplace=True)

    # drop any rows in the dataset that have NaNs
    df.dropna(how="any")

    # create categorical columns
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].astype('category')

    df.drop('status_group')

    return pd.get_dummies(df)


# # Loading local development files
# 
# If I'm just loading local python files that I expect to use in this project, I often just add the `src` folder to the Python path using `sys.path.append`. This tells Python to look in that folder for modules that we can import. This works well for local code and notebooks.

# In[18]:

# Load the "autoreload" extension
get_ipython().magic('load_ext autoreload')

# always reload modules marked with "%aimport"
get_ipython().magic('autoreload 1')

import os
import sys

# add the 'src' directory as one where we can import modules
src_dir = os.path.join(os.getcwd(), os.pardir, 'src')
sys.path.append(src_dir)

# import my method from the source code
get_ipython().magic('aimport features.build_features')
from features.build_features import remove_invalid_data
from features.build_features import awesome_function


# In[19]:

# edit function in file!
awesome_function("ODSC")


# In[23]:

df = remove_invalid_data(pump_data_path)


# As mentioned in the slides, using `sys.path.append` is not the best way to distribute code that you want to run on other machines. For that, create a real Python package that can be separately developed, maintained, and deployed.
# 
# We can build a python package to solve that! In fact, there is a cookiecutter to create Python packages.
# Once we create this package, we can install it in "editable" mode, which means that as we change the code the changes will get picked up if the package is used. The process looks like:
# 
#     cookiecutter https://github.com/wdm0006/cookiecutter-pipproject
#     cd package_name
#     pip install -e .
# 
# Now we can have a separate repository for this code and it can be used across projects without having to maintain code in multiple places.

# -------------------------
# 
# ------------------------

# # Let's train a model!
# 
# Now, we'll use `sklearn` to train a machine learning model. We'll just do a simple logistic regression model, and for fun we'll use `PolynomialFeatures` to generate interaction terms. 
# 
# 
# #### #lifehack: if something goes wrong use `%debug` !

# In[29]:

get_ipython().run_cell_magic('prun', '', "\nfrom sklearn.pipeline import Pipeline\n\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.preprocessing import PolynomialFeatures\n\ndf = remove_invalid_data(pump_data_path)\n\nlabels = (pd.read_csv(os.path.join(PROJ_ROOT, 'data', 'raw', 'pumps_train_labels.csv'),\n                     index_col=0)\n            .loc[df.index])\n\npl = Pipeline([\n    ('interactions', PolynomialFeatures(degree=2)),\n    ('clf', LogisticRegression())\n])\n\npl.fit(df, labels)")


# In[30]:

pl.predict(df)


# #### #lifehack: if something takes a long time use `%%prun` !

# In[31]:

import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig(os.path.join(PROJ_ROOT, "reports", "figures", "confusion_matrix.png"))
    plt.show()


# In[33]:

cm = confusion_matrix(labels, pl.predict(df),
                 labels=['functional', 'non functional', 'functional needs repair'])

plot_confusion_matrix(cm,
                      ['functional', 'non functional', 'functional needs repair'])


# # Now let's see what we've put together!

# In[35]:

get_ipython().system('tree {PROJ_ROOT}/data')


# In[ ]:



