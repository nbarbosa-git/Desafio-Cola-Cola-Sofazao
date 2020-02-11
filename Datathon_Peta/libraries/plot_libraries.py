
##########
# File: plot_libraries.py
# Description:
#    Coleção de funções para criar gráficos
##########


import matplotlib.pyplot as plt
import seaborn as sns


#configuracao basica dos graficos
def setup_graphics():
  # Plotting options


  sns.set(style='whitegrid')
  plt.rcParams['savefig.dpi'] = 75
  plt.rcParams['figure.autolayout'] = False
  plt.rcParams['figure.figsize'] = 10, 6
  plt.rcParams['axes.labelsize'] = 18
  plt.rcParams['axes.titlesize'] = 20
  plt.rcParams['font.size'] = 16
  plt.rcParams['lines.linewidth'] = 2.0
  plt.rcParams['lines.markersize'] = 8
  plt.rcParams['legend.fontsize'] = 14




#plot graficos EDA
  def plot_var(df, y_axis, stack, x_axis1, x_axis2, agg='Mean'):

    #-----------------------------
    plt.rcParams['savefig.dpi'] = 75
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.figsize'] = 10, 6
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['font.size'] = 8
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    #-----------------------------


    #SETUP GRAPHIC SPACE (1X2)
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    cmap = plt.cm.inferno


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

    plt.tight_layout()
    return data
    #return df.groupby(x_axis2)[y_axis].describe()