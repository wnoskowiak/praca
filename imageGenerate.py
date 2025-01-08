import numpy as np
import matplotlib.pyplot as plt
from microboone_utils import *
from skimage.measure import block_reduce

from pynuml.io import File
f = File("../Inclusive_with_wire_info/bnb_WithWire_00.h5")

tables = ['event_table','wire_table','hit_table','edep_table']
for t in tables: f.add_group(t)
# print(t)
f.read_data(0, 8)
# print(t)
evts = f.build_evt()

for evt in evts:
    # print(len(evts))
    evt_id = [evt["event_table"]["run"].iloc[0],evt["event_table"]["subrun"].iloc[0],evt["event_table"]["event"].iloc[0]]
    # print('Going to produce wire image for event:',evt_id)

    wires = evt["wire_table"]
    planeadcs = [wires.query("local_plane==%i"%p)[['adc_%i'%i for i in range(0,ntimeticks())]].to_numpy() for p in range(0,nplanes())]

    aspratio = len(planeadcs[2])/len(planeadcs[1])

    f_downsample = 6
    for p in range(0,nplanes()):
        planeadcs[p] = block_reduce(planeadcs[p], block_size=(1,f_downsample), func=np.sum)
        
    adcsgt0 = [i[i>0] for i in planeadcs[p] for p in range(0,nplanes())]
    flat_adcs = [item for sublist in adcsgt0 for item in sublist]

    adccutoff = 10.*f_downsample/6.
    adcsaturation = 100.*f_downsample/6.
    for p in range(0,nplanes()):
        planeadcs[p][planeadcs[p]<adccutoff] = 0
        planeadcs[p][planeadcs[p]>adcsaturation] = adcsaturation
        
    # graph saving

    zmax = adcsaturation

    print("Run / Sub / Event : %i / %i / %i - saturation set to ADC sum=%.2f"%(evt_id[0],evt_id[1],evt_id[2],zmax))

    # Plot and save Plane 0
    fig, ax1 = plt.subplots(figsize=(20, 5), dpi=600)
    im1 = ax1.imshow(planeadcs[0].T, vmin=0, vmax=zmax, origin='lower', cmap='jet')
    ax1.set_title("Plane 0")
    ax1.set_xlabel("Wire")
    ax1.set_ylabel("Time Tick")
    plt.tight_layout()
    plt.savefig(f"{evt_id}plane_0.png")
    plt.close(fig)

    # Plot and save Plane 1
    fig, ax2 = plt.subplots(figsize=(20, 5), dpi=600)
    im2 = ax2.imshow(planeadcs[1].T, vmin=0, vmax=zmax, origin='lower', cmap='jet')
    ax2.set_title("Plane 1")
    ax2.set_xlabel("Wire")
    plt.tight_layout()
    plt.savefig(f"{evt_id}plane_1.png")
    plt.close(fig)

    # Plot and save Plane 2
    fig, ax3 = plt.subplots(figsize=(20, 5), dpi=600)
    im3 = ax3.imshow(planeadcs[2].T, vmin=0, vmax=zmax, origin='lower', cmap='jet')
    ax3.set_title("Plane 2")
    ax3.set_xlabel("Wire")
    plt.tight_layout()
    plt.savefig(f"{evt_id}plane_2.png")
    plt.close(fig)
        
        
    # label 

    hits = evt["hit_table"]
    edeps = evt["edep_table"]
    edeps = edeps.sort_values(by=['energy_fraction'], ascending=False, kind='mergesort').drop_duplicates(["hit_id"])
    hits = hits.merge(edeps, on=["hit_id"], how="left")
    hits['g4_id'] = hits['g4_id'].fillna(-1)
    hits = hits.fillna(0)

    print(len(hits))