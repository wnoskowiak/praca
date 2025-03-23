from skimage.measure import block_reduce
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from pynuml.io import File
import pandas as pd
from microboone_utils import *

def get_evt_id(evt):
    return [evt["event_table"][key].values[0] for key in ["run","subrun","event"]]


def process_event(wires, evt_id, save_dir):
    
    def get_file_names(evt_id):
        return [f'{evt_id[0]}_{evt_id[1]}_{evt_id[2]}_{i}.png' for i in range(3)]
    
    planeadcs = [wires.query("local_plane==%i"%p)[['adc_%i'%i for i in range(0,ntimeticks())]].to_numpy() for p in range(0,nplanes())]

    f_downsample = 6
    for p in range(0,nplanes()):
        planeadcs[p] = block_reduce(planeadcs[p], block_size=(1,f_downsample), func=np.sum)

    adccutoff = 10.*f_downsample/6.
    adcsaturation = 100.*f_downsample/6.
    for p in range(0,nplanes()):
        planeadcs[p][planeadcs[p]<adccutoff] = 0
        planeadcs[p][planeadcs[p]>adcsaturation] = adcsaturation

    zmax = adcsaturation
    
    norm = plt.Normalize(vmin=0, vmax=zmax)
    cmap = plt.cm.Greys
    file_names = get_file_names(evt_id)
    
    for i in range(3):
        plt.imsave(os.path.join(save_dir, file_names[i]), cmap(norm(planeadcs[i].T)))
        
    return file_names

inputDir = "/workspace/Inclusive_with_wire_info"
outputDir = "/workspace/outputDir"

files = os.listdir(inputDir)
h5_files = [file for file in files if file.endswith('.h5')]

print(h5_files)

columnsToRead = [
    ['event_table', 'is_cc'],
    ['event_table', 'nu_energy'],
    ['event_table', 'nu_pdg'],
]

batchSize = 16
csv_file_path = os.path.join(outputDir, 'parsed_data_dos.csv')


with open(csv_file_path, 'w') as csv_file:
    csv_file.write('run,subrun,event,is_cc,nu_energy,nu_pdg,plane0_file,plane1_file,plane2_file\n')

for file in h5_files:
    file_path = os.path.join(inputDir, file)
    print(f"Processing {file_path}")

    f = File(file_path)
    tables = ['event_table', 'wire_table']
    for t in tables:
        f.add_group(t)
    evtNum = len(f)

    for start in range(0, evtNum, batchSize):
        print(f"Processing events {start} - {min(start + batchSize, evtNum)}")
        
        cout = min(batchSize, evtNum - start)
        f.read_data(start, cout)
        evts = f.build_evt(start, cout)

        with open(csv_file_path, 'a') as csv_file:
            for evt in evts:
                try:
                    evt_id = get_evt_id(evt)
                    print(f"Processing event {evt_id}")
                    data = [evt[table][column].values[0] for table, column in columnsToRead]
                    files = process_event(evt['wire_table'], evt_id, outputDir)
                    row = evt_id + data + files
                    csv_file.write(','.join(map(str, row)) + '\n')

                except Exception as e:
                    print(f"Failed to process event {evt_id}: {e}")

