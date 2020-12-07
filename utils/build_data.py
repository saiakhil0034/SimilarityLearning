import numpy as np
import pandas as pd
import os

def get_data(data_path, seqs):

    data_frames = []
    for i in seqs:
        seq_path = f'000{i}' if i < 10 else f'00{i}'
        dir_path = f'{data_path}/{seq_path}'
        objects = os.listdir(dir_path)
        for object_id in objects:
            od_path = dir_path + f"/{object_id}"
            frames = os.listdir(od_path)
            label = object_id
            num_frames = len(frames)
            if num_frames > 3:
                paths = np.array([od_path + f"/{f}" for f in frames if f[-4:]
                                  in ('.npy', '.npz')])
                ob_labels = [label for _ in paths]
                obj_data = pd.DataFrame()
                obj_data["path"] = paths
                obj_data["label"] = np.array(ob_labels)
                # print(obj_data.shape)
                data_frames.append(obj_data)
    data = pd.concat(data_frames)
    return data
