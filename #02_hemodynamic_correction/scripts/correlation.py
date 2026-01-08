import os
from os.path import join as pjoin
from typing import List, Union, Tuple
from tifffile import *
from tqdm import tqdm

import numpy as np
from scipy.stats import pearsonr

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def get_selector(id_arr:np.ndarray, qid_list=None):
    if qid_list is None:
        return np.ones(shape=id_arr.shape).astype(bool)
    else:
        selector = np.zeros(shape=id_arr.shape)
        for id in qid_list:
            bool_arr = id == id_arr
            selector += np.array(bool_arr).astype(int)
        return selector.astype(bool)

def make_heatmap(tc_matrix1:np.ndarray, tc_matrix2:np.ndarray)->np.ndarray:
    # tc_matrix ; (n_components, NFrame)
    heatmap = np.empty((int(len(tc_matrix1)), int(len(tc_matrix2))))
    for index_i, i in enumerate(tc_matrix1):
        for index_j, j in enumerate(tc_matrix2):
            correlation_coefficient, _ = pearsonr(i, j)
            heatmap[index_i,index_j] = correlation_coefficient
    return heatmap

def filter_heatmap(heatmap:np.ndarray, threshhold:np.float16)->np.ndarray:
    if abs(threshhold) > 1.0:
        raise ValueError("Threshhold should be between -1.0 to 1.0")
    heatmap_filterd = heatmap.copy()
    heatmap_filterd[abs(heatmap_filterd) < abs(threshhold)] = 0
    return heatmap_filterd

def show_each_traces(traces:np.ndarray, mylabel:list, 
                     sf:float = 30, plotting_sf:int = 10, offset:int=5,
                     extend:int = 1.0, magnify:int = 1.0, time_bar_data_coord:int = 60, 
                     title:str="temporal_components", save_path:str=None)->None:
    # make time series
    length = len(traces[0])
    time = np.arange(0,length) / sf

    # adding offset is for visualization
    traces_with_offset = []
    num_traces = len(traces)
    for trace, i in zip(traces, range(num_traces)):
        _offset = i*offset
        traces_with_offset.append(trace + _offset)

    # make instances
    plt.rcParams['figure.figsize'] = [extend*magnify*8, magnify*6]
    fig, ax = plt.subplots()
    plt.title(title)

    # secure the margins so that sacle bars are going to be well embeded.
    ax.set_xlim(0, 1.10*np.max(time))
    ax.set_ylim(-1.0, np.max(np.vstack(traces_with_offset)))


    for label, trace in zip(mylabel, traces_with_offset):

        ## base plot
        ax.plot(time[::plotting_sf], trace[::plotting_sf], color = "black")

        # 上下の余白を調整
        plt.subplots_adjust(top=0.7, bottom=0.1)

        ## removing the spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ## removing ticks and labels
        ax.tick_params(axis='both', which='both', bottom=False, left=False, 
                labelbottom=False, labelleft=False)

        ## Add the count as text
        ## Add the count as text on the left of the trace
        if extend >= 3.0:
            ax.text(time[0] - 0.01*(time[-1] - time[0]), trace[0], label,
                    verticalalignment='center', color='black', fontsize=10)
        else:
            ax.text(time[0] - 0.05*(time[-1] - time[0]), trace[0], label,
                    verticalalignment='center', color='black', fontsize=10)


    hline_endposition_x_data_coord = max(time)+0.05*max(time)
    end_trace_sys_coord = 1.0 * (hline_endposition_x_data_coord/ax.get_xlim()[1])
    _time_bar_data_coord = time_bar_data_coord
    time_bar_sys_coord = 1.0 * (_time_bar_data_coord/ax.get_xlim()[1])
    hline_position_y_data_coord = -0.5
    ax.axhline(y=hline_position_y_data_coord,
            xmin=end_trace_sys_coord-time_bar_sys_coord,
            xmax=end_trace_sys_coord,
            color="black", linewidth = 2)

    val_bar_data_coord = 0.5
    val_bar_sys_coord = 1.0 * (val_bar_data_coord/(abs(ax.get_ylim()[0])+ax.get_ylim()[1]))
    # hline_position_y_sys_coord = abs(ax.get_ylim()[0]-hline_position_y_data_coord)/(abs(ax.get_ylim()[0])+ax.get_ylim()[1])
    # ax.axvline(x=hline_endposition_x_data_coord,
    #         ymin=hline_position_y_sys_coord,
    #         ymax=hline_position_y_sys_coord+val_bar_sys_coord,
    #         color="black", linewidth = 2)

    htext_position_x_data_coord = hline_endposition_x_data_coord-_time_bar_data_coord
    if num_traces <= 10:
        htext_position_y_data_coord = + (np.max(np.vstack(traces_with_offset))/len(mylabel))/5
    else:
        htext_position_y_data_coord = ax.get_ylim()[0] - 0.5
    ax.text(x = htext_position_x_data_coord,
            y = htext_position_y_data_coord,
            s = f"{_time_bar_data_coord} sec")

    # vtext_position_x_data_coord = hline_endposition_x_data_coord + 0.1 * (ax.get_xlim()[1]-hline_endposition_x_data_coord)
    # vtext_position_y_data_coord = hline_position_y_data_coord + 0.5 * (val_bar_data_coord)
    # ax.text(x=vtext_position_x_data_coord,
    #         y=vtext_position_y_data_coord,
    #         s= r'$\Delta$' + f"F={int(val_bar_data_coord*100)}%")

    # plt.title("Traces of temporal components")
    
    if save_path is not None:
        fig.savefig(save_path)

    plt.show()

def show_overlaid_traces(traces:np.ndarray, mylabel:list, sf = 30, 
                         plotting_sf:float = 10, magnify:int = 1.0, time_bar_data_coord:int=60, 
                         title:str="temporal_components", save_path:str=None)->None:
    # make time series
    length = len(traces[0])
    time = np.arange(0,length) / sf
    num_traces = len(traces)

    # make instances
    plt.rcParams['figure.figsize'] = [3*magnify*8, magnify*6]
    fig, ax = plt.subplots()

    plt.title(title)

    # secure the margins so that sacle bars are going to be well embeded.
    ax.set_xlim(0, 1.10*np.max(time))
    ax.set_ylim(-1.0, np.max(np.vstack(traces)))


    for label, trace in zip(mylabel, traces):

        ax.plot(time[::plotting_sf], trace[::plotting_sf], label = label)
        ax.legend()

        # 上下の余白を調整
        plt.subplots_adjust(top=0.7, bottom=0.1)

        ## removing the spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ## removing ticks and labels
        ax.tick_params(axis='both', which='both', bottom=False, left=False, 
                labelbottom=False, labelleft=False)

        # ## Add the count as text
        # ## Add the count as text on the left of the trace
        # ax.text(time[0] - 0.1*(time[-1] - time[0]), trace[0], label,
        #         verticalalignment='center', color='black', fontsize=10)


    hline_endposition_x_data_coord = max(time)+0.05*max(time)
    end_trace_sys_coord = 1.0 * (hline_endposition_x_data_coord/ax.get_xlim()[1])
    _time_bar_data_coord = time_bar_data_coord
    time_bar_sys_coord = 1.0 * (_time_bar_data_coord/ax.get_xlim()[1])
    hline_position_y_data_coord = -0.5
    ax.axhline(y=hline_position_y_data_coord,
            xmin=end_trace_sys_coord-time_bar_sys_coord,
            xmax=end_trace_sys_coord,
            color="black", linewidth = 2)

    val_bar_data_coord = 0.5
    val_bar_sys_coord = 1.0 * (val_bar_data_coord/(abs(ax.get_ylim()[0])+ax.get_ylim()[1]))
    # hline_position_y_sys_coord = abs(ax.get_ylim()[0]-hline_position_y_data_coord)/(abs(ax.get_ylim()[0])+ax.get_ylim()[1])
    # ax.axvline(x=hline_endposition_x_data_coord,
    #         ymin=hline_position_y_sys_coord,
    #         ymax=hline_position_y_sys_coord+val_bar_sys_coord,
    #         color="black", linewidth = 2)

    htext_position_x_data_coord = hline_endposition_x_data_coord-_time_bar_data_coord
    if num_traces <= 10:
        htext_position_y_data_coord = + (np.max(np.vstack(traces))/len(mylabel))/5
    else:
        htext_position_y_data_coord = ax.get_ylim()[0] - 0.5
    ax.text(x = htext_position_x_data_coord,
            y = htext_position_y_data_coord,
            s = f"{_time_bar_data_coord} sec")

    # vtext_position_x_data_coord = hline_endposition_x_data_coord + 0.1 * (ax.get_xlim()[1]-hline_endposition_x_data_coord)
    # vtext_position_y_data_coord = hline_position_y_data_coord + 0.5 * (val_bar_data_coord)
    # ax.text(x=vtext_position_x_data_coord,
    #         y=vtext_position_y_data_coord,
    #         s= r'$\Delta$' + f"F={int(val_bar_data_coord*100)}%")

    # plt.title("Traces of temporal components")
    
    if save_path is not None:
        fig.savefig(save_path)

    plt.show()

def show_heatmap(heatmap:np.ndarray, xlabel=None, ylabel=None, ax=None, magnify = 1.0, title="Heat Map", save_path=None)->None:
    
    row, col = heatmap.shape
    if row == col:
        plt.rcParams['figure.figsize'] = [magnify*8, magnify*6]
    else:
        plt.rcParams['figure.figsize'] = [(col/100)*magnify*8, (row/100)*magnify*6]

    if ax is None:
        ax = plt.gca()

    # fig.suptitle("Heatmap of Pearson correlation between each Ics", fontsize=15)
    im = ax.imshow(heatmap, cmap='coolwarm', vmin=-1, vmax=1)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")
    ax.xaxis.set_ticks_position('top')

    # Show all ticks and label them with the respective list entries
    if xlabel is not None:
        ax.set_xticks(np.arange(len(xlabel)), labels=xlabel)
    else:
        # ax.xaxis.set_label_coords(0.5, 1.20)
        for label in ax.get_xticklabels():
            label.set_visible(False)
    
    if ylabel is not None:
        ax.set_yticks(np.arange(len(ylabel)), labels=ylabel)
    else:
        for label in ax.get_yticklabels():
            label.set_visible(False)   

    ax.set_title(title)

    if row <= 10 and col <= 10:
        heatmap_for_annotation = np.round(heatmap, decimals=4)
        for i in range(row):
            for j in range(col):
                text = ax.text(j, i, heatmap_for_annotation[i, j],
                            ha="center", va="center", color="w")


    if save_path is not None:
        fig = plt.gcf()
        fig.savefig(save_path)
        
    plt.show()


    
def get_cluster_index(tree, path_route:List[str])->Union[int, int]:
    '''
    INPUT
        - tree : scipy.cluster.hierarchy.ClusterNode object that is returned as below
        Z = heatmap_cluster.dendrogram_row.linkage
        tree = to_tree(Z)

        - List : List of python string object for decending tree structure 
       
    OUTPUT
        - Tuple : (index number for start, index number for end)

    Usage
        - reordered_ind = heatmap_cluster.dendrogram_row.reordered_ind
        reordered_ind[start:end]

    '''

    # initialize the start and end with a single whole cluster
    valid_characters = {"L", "l", "R", "r"}
    if any(char not in valid_characters for char in path_route):
        raise ValueError("Error : Only 'L', 'l', 'R', 'r' are valid in path_route argument")

    start = 0
    end = tree.get_count()

    def _ordinal(number):
        if 10 <= number % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
        
        return f"{number}{suffix}"

    for loop, i in enumerate(path_route, start=1):
        if i in ["L", "l"]:
            new_tree = tree.get_left()
            end -= (tree.get_right()).get_count()
            tree = new_tree
        if i in ["R", "r"]:
            new_tree = tree.get_right()
            start = end - new_tree.get_count()
            tree = new_tree
    
        print(f"After {_ordinal(loop)} choice ; {i}")
        print(f"start:{start}")
        print(f"end:{end}")

    return (start, end)
    

## below are example code

# path_to_route = ['l', 'l', 'l']
# start, end = get_cluster_index(tree, path_to_route)
# print(f"final result")
# print(f"start:{start}")
# print(f"end:{end}")