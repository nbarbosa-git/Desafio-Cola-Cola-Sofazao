
##########
# File: plot_libraries.py
# Description:
#    Coleção de funções para criar gráficos!
##########


from matplotlib import pyplot
import seaborn as sns
import pandas as pd


#configuracao basica dos graficos
def setup_graphics():
  # Plotting options


  sns.set(style='whitegrid')
  pyplot.rcParams['savefig.dpi'] = 75
  pyplot.rcParams['figure.autolayout'] = False
  pyplot.rcParams['figure.figsize'] = 10, 6
  pyplot.rcParams['axes.labelsize'] = 18
  pyplot.rcParams['axes.titlesize'] = 20
  pyplot.rcParams['font.size'] = 16
  pyplot.rcParams['lines.linewidth'] = 2.0
  pyplot.rcParams['lines.markersize'] = 8
  pyplot.rcParams['legend.fontsize'] = 14


#verificar residuo gaussiano
def plot_residuos(X, Xt):

  fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)
  #fig.suptitle('Sharing x per column, y per row')

  X.hist(ax=ax1) 
  Xt.hist(ax=ax2, color='coral') 

  X.plot(kind='kde', ax=ax3)
  Xt.plot(kind='kde', ax=ax4, color='coral')
  
  ax1.set_title("Antes")
  ax2.set_title("Depois")

  #for ax in fig.get_axes(): ax.label_outer()
  pyplot.tight_layout()
  pyplot.show()

  #some statistics
  stat_X = X.groupby(X.index.year).agg(['mean', 'std']).T
  stat_Xt = Xt.groupby(Xt.index.year).agg(['mean', 'std']).T
  return stat_Xt #, stat_Xt



  #verificar residuo gaussiano
def plot_error(X, Xt):

  fig, ((ax1, ax2), (ax3, ax4)) = pyplot.subplots(2, 2)
  #fig.suptitle('Sharing x per column, y per row')

  X.hist(ax=ax1) 
  Xt.hist(ax=ax2, color='coral') 

  X.plot(kind='kde', ax=ax3)
  Xt.plot(kind='kde', ax=ax4, color='coral')
  
  ax1.set_title("svr")
  ax2.set_title("lgbm")

  pyplot.tight_layout()
  pyplot.show()


#plot graficos EDA
def plot_var(df, y_axis, stack, x_axis1, x_axis2, agg='Mean'):

    #-----------------------------
    sns.set(style='whitegrid')
    pyplot.rcParams['savefig.dpi'] = 75
    pyplot.rcParams['figure.autolayout'] = False
    pyplot.rcParams['figure.figsize'] = 10, 6
    pyplot.rcParams['axes.labelsize'] = 8
    pyplot.rcParams['axes.titlesize'] = 12
    pyplot.rcParams['font.size'] = 8
    pyplot.rcParams['lines.linewidth'] = 2.5
    pyplot.rcParams['lines.markersize'] = 8
    pyplot.rcParams['legend.fontsize'] = 8
    pyplot.rcParams['xtick.labelsize'] = 8
    pyplot.rcParams['ytick.labelsize'] = 8
    #-----------------------------


    #SETUP GRAPHIC SPACE (1X2)
    f, (ax1, ax2) = pyplot.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    cmap = pyplot.cm.inferno


    #DATA
    data = df.groupby([x_axis1, stack])

    #Aggregate by mean
    if agg == 'Mean':
       data = data[y_axis].mean().unstack()

    #Aggregate by sum
    else: data = data[y_axis].sum().unstack()
    

    #PLOT GRAPHIC 1
    data.plot(kind='area', stacked=True,
                        colormap=cmap, grid=False, 
                        legend=False, ax=ax1,
                        figsize=(12,3))
    

    #SETUP GRAPHIC 1
    ax1.set_title(agg +' of ' + y_axis + ' by ' + stack)
    ax1.set_xlabel(x_axis1)
    ax1.set_ylabel(y_axis)

    ax1.legend(bbox_to_anchor=(0.2, 0.9, 0.6, 0.1), 
               loc=10, prop={'size':7},
               ncol=len(list(data.columns)),
               mode="expand", borderaxespad=0.0)
  

#-----------------------------

    #PLOT GRAPHIC 2
    sns.boxplot(y=y_axis, x=x_axis2, data=df, ax=ax2)


    #SETUP GRAPHIC 2
    ax2.set_ylabel('')
    ax2.set_title(y_axis + ' by ' + x_axis2)
    ax2.set_xlabel(x_axis2)

    pyplot.tight_layout()
    return data
    #return df.groupby(x_axis2)[y_axis].describe()