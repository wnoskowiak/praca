import numpy as np
import matplotlib.pyplot as plt
from microboone_utils import *
from skimage.measure import block_reduce
import sqlite3
import os
from io import BytesIO
from PIL import Image

from pynuml.io import File

# Read file path from environment variable
file_path = os.getenv('H5_FILE_PATH', '../Inclusive_with_wire_info/bnb_WithWire_00.h5')
f = File(file_path)

tables = ['event_table','wire_table','hit_table','edep_table']
for t in tables: f.add_group(t)
f.read_data(0, 8)
evts = f.build_evt()

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('event_data.db')
c = conn.cursor()

# Create table for images and labels
c.execute('''
    CREATE TABLE IF NOT EXISTS event_data (
        event_id TEXT,
        plane INTEGER,
        image BLOB,
        hit_id INTEGER,
        g4_id INTEGER,
        energy_fraction REAL,
        label INTEGER,
        PRIMARY KEY (event_id, plane, hit_id)
    )
''')

for evt in evts:
    evt_id = "%i_%i_%i" % (evt["event_table"]["run"].iloc[0], evt["event_table"]["subrun"].iloc[0], evt["event_table"]["event"].iloc[0])

    wires = evt["wire_table"]
    planeadcs = [wires.query("local_plane==%i" % p)[['adc_%i' % i for i in range(0, ntimeticks())]].to_numpy() for p in range(0, nplanes())]

    f_downsample = 6
    for p in range(0, nplanes()):
        planeadcs[p] = block_reduce(planeadcs[p], block_size=(1, f_downsample), func=np.sum)

    adccutoff = 10. * f_downsample / 6.
    adcsaturation = 100. * f_downsample / 6.
    for p in range(0, nplanes()):
        planeadcs[p][planeadcs[p] < adccutoff] = 0
        planeadcs[p][planeadcs[p] > adcsaturation] = adcsaturation

    zmax = adcsaturation

    print("Run / Sub / Event : %i / %i / %i - saturation set to ADC sum=%.2f" % (evt_id[0], evt_id[1], evt_id[2], zmax))

    # Function to save image to database
    def save_image_to_db(event_id, plane, image_array, label):
        img = Image.fromarray((image_array * 255 / zmax).astype(np.uint8))
        with BytesIO() as output:
            img.save(output, format="PNG")
            image_blob = output.getvalue()
        c.execute("INSERT OR REPLACE INTO event_data (event_id, plane, image, label) VALUES (?, ?, ?, ?)", (event_id, plane, image_blob, label))

    # Save labels
    hits = evt["hit_table"]
    edeps = evt["edep_table"]
    edeps = edeps.sort_values(by=['energy_fraction'], ascending=False, kind='mergesort').drop_duplicates(["hit_id"])
    hits = hits.merge(edeps, on=["hit_id"], how="left")
    hits['g4_id'] = hits['g4_id'].fillna(-1)
    hits = hits.fillna(0)

    label = 1 if len(hits) > 0 else 0

    # Save Plane 0 image
    save_image_to_db(evt_id, 0, planeadcs[0].T, label)

    # Save Plane 1 image
    save_image_to_db(evt_id, 1, planeadcs[1].T, label)

    # Save Plane 2 image
    save_image_to_db(evt_id, 2, planeadcs[2].T, label)

    for _, hit in hits.iterrows():
        c.execute("INSERT OR REPLACE INTO event_data (event_id, plane, image, hit_id, g4_id, energy_fraction, label) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (evt_id, hit['local_plane'], None, hit['hit_id'], hit['g4_id'], hit['energy_fraction'], label))

    print(len(hits))

# Commit and close the database connection
conn.commit()
conn.close()