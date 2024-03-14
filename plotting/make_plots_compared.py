import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


"""
A method for comparasion the tables obtained in R and Python. 
""" 
def make_plots_compared(roc_matrix_breast_p, roc_matrix_breast_R, legend_loc, title, models_labels):
    """
    Takes two data frames and plots a boxplot of them on one figure

    Parameters
    ----------
    roc_matrix_breast_p : data frame
        The matrix obtained using Python

    roc_matrix_breast_python : data frame
        The matrix obtained using Python

    legend_loc: string
        Position of the legend. Takes values 'topleft', 'topright', 'bottomleft', 'bottomright' and 'best'.

    title: string
        Title of the plot

    models_labels: list
        labels for each box plot


    Returns
    -------
    The boxplot of the given data frames
    """

    roc_matrix_breast_p.columns = models_labels
    roc_matrix_breast_R.columns = models_labels
    cdf = pd.concat([roc_matrix_breast_p.assign(Software = 'Python'), roc_matrix_breast_R.assign(Software = 'R')])
    mdf = pd.melt(cdf, id_vars=['Software'], var_name='Model').rename(columns = {'value': 'AUC score'})
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    b = sns.boxplot(x="Model", y="AUC score", hue="Software", fill=True, data=mdf)
    b.set_xlabel("Model", fontsize=20)
    b.set_ylabel("AUC", fontsize=20)
    b.tick_params(labelsize=16)
    plt.title(title,  fontsize = 30)
    plt.legend(loc=legend_loc, fontsize="28")
    plt.subplots_adjust(left=0.085, right=0.99, top=0.99, bottom=0.1)
    plt.show()
    
    print('plotting is done!')

