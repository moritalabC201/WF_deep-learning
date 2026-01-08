from typing import List, Tuple
import numpy as np

def get_indexes_for_each_trial(trial_length:List)->List[Tuple]:

    start = [0]
    for i in range(len(trial_length)):
        start.append(start[-1]+trial_length[i])

    trial_indexes = []
    for start_index, length in zip(start, trial_length):
        trial_indexes.append((start_index, start_index+length))

    return trial_indexes


def extract_links_from_heatmap(heatmap:np.ndarray)->list:
    
    return_llink_list = []
    for index, i in zip(range(len(heatmap)), heatmap):
        return_llink_list.extend(i[index:])
    
    return return_llink_list