import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

def prediction_all_samples(model, dataloader):
    model.eval()
    labels = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, label = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            labels.extend(label.tolist())
    return pd.DataFrame( {"y_pred": predictions, "y_true": labels})

def error_analysis(df):
    df = df.copy()
    df["error"] = df["y_pred"] != df["y_true"]
    df["error"] = df["error"].astype(int)
    return df

def plot_bargraph_with_groupings(df, groupby, colourby, title, xlabel, ylabel):
    """
    Plots a dataframe showing the frequency of datapoints grouped by one column and coloured by another.
    df : dataframe
    groupby: the column to groupby
    colourby: the column to color by
    title: the graph title
    xlabel: the x label,
    ylabel: the y label
    """
    # Makes a mapping from the unique colourby column items to a random color.
    ind_col_map = {x:y for x, y in zip(df[colourby].unique(),
                               [plt.cm.Paired(np.arange(len(df[colourby].unique())))][0])}


    # Find when the indicies of the soon to be bar graphs colors.
    unique_comb = df[[groupby, colourby]].drop_duplicates()
    name_ind_map = {x:y for x, y in zip(unique_comb[groupby], unique_comb[colourby])}
    c = df[groupby].value_counts().index.map(lambda x: ind_col_map[name_ind_map[x]])

    # Makes the bargraph.
    ax = df[groupby].value_counts().plot(kind='bar',
                                         figsize=FIG_SIZE,
                                         title=title,
                                         color=[c.values])
    # Makes a legend using the ind_col_map
    legend_list = []
    for key in ind_col_map.keys():
        legend_list.append(mpatches.Patch(color=ind_col_map[key], label=key))

    # display the graph.
    plt.legend(handles=legend_list)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)