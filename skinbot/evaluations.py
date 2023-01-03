import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

def predict_samples(model, dataloader, fold, target_mode, N=None):
    model.eval()
    labels = []
    predictions = []
    probs = []
    is_fuzzy = target_mode in ["fuzzy", "multi"]
    labels_fuzzy = []
    with torch.no_grad():
        for batch in dataloader:
            inputs, label = batch
            outputs = model(inputs)
            prob = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)
            predictions.extend(predicted.tolist())
            # if fuzzy we need to compute the index as the max value
            if is_fuzzy:
                labels_fuzzy.extend(label.tolist())
                label = torch.argmax(label, dim=1) if len(label.shape) > 1 else label
            labels.extend(label.tolist())
            probs.extend(prob.tolist())
            cnt = len(labels)
            if N is not None:
                logging.info('pred: ', cnt, 'out of ', N)
                if cnt > N:
                    break
    if is_fuzzy:
        return pd.DataFrame( {"y_pred": predictions, "y_true": labels, "y_prob": probs, "y_true_fuzzy": labels_fuzzy,
                              'files': dataloader.dataset.image_fnames,
                              'fold': len(predictions)*[fold]})
    else:
        return pd.DataFrame( {"y_pred": predictions, "y_true": labels, "y_prob": probs})

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