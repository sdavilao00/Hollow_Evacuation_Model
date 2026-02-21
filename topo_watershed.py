# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 16:28:04 2025

@author: sdavilao
"""

import topotoolbox as tt3
import matplotlib.pyplot as plt

dem = tt3.read_tif("full_extent_crop1_32610_10m.tif")
fd = tt3.FlowObject(dem)


acc = fd.flow_accumulation()
im = acc.plot(cmap='Blues',norm="log")
plt.colorbar(im)
plt.show()

D = fd.drainagebasins()
D.shufflelabel().plot(cmap="Pastel1",interpolation="nearest")
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import xy


# 0) Load + flow
dem = tt3.read_tif("full_extent_crop1_32610_10m.tif")
fd  = tt3.FlowObject(dem)
acc = fd.flow_accumulation()              # upstream contributing *cells*
acc_cells = acc.z                         # IMPORTANT: cells, not m²

# 1) Initial basins to discover exterior outlets
D0 = fd.drainagebasins()
labels0 = D0.z.astype(np.int32)
nlabels = labels0.max()

# 2) Make an "edge" mask (exterior contact)
edge = np.zeros_like(labels0, dtype=bool)
edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True

# 3) For each basin label, find its outlet on the edge (edge pixel with max acc)
#    We'll store (label, row, col, acc_at_outlet)
outlets = []
for lab in range(1, nlabels + 1):
    on_edge = edge & (labels0 == lab)
    if not np.any(on_edge):
        continue  # interior basin (shouldn't happen if sinks are handled), skip
    # pick the edge pixel with the largest contributing cells = the pour point
    idx = np.argmax(acc_cells[on_edge])
    rows, cols = np.where(on_edge)
    r, c = rows[idx], cols[idx]
    outlets.append((lab, r, c, acc_cells[r, c]))

# 4) Apply GRASS-like threshold: keep only outlets with acc >= MIN_AREA_CELLS
#    Choose in *cells*, not m². If you want m², convert: cells = area_m2 / (cellsize**2)
MIN_AREA_CELLS = 40000 // 100  # example: 40,000 m² with 10 m cells -> 400 cells
# or explicitly: MIN_AREA_CELLS = int(np.ceil(40000.0 / (dem.cellsize**2)))

kept = [(lab, r, c, a) for (lab, r, c, a) in outlets if a >= MIN_AREA_CELLS]

print(f"Initial basins: {nlabels} | Exterior outlets found: {len(outlets)} | Kept seeds: {len(kept)}")
if len(kept) == 0:
    print("⚠️ No outlets meet the threshold. Lower MIN_AREA_CELLS or confirm units.")
    
# 5) Build seed points from the kept outlets
#    Option A: seeds as GeoDataFrame of points (if your tt3 supports seeds=points)
xs, ys, ids = [], [], []
for i, (lab, r, c, a) in enumerate(kept, start=1):
    x, y = xy(dem.transform, r, c)
    xs.append(x); ys.append(y); ids.append(i)

seeds_gdf = gpd.GeoDataFrame({"seed_id": ids}, geometry=gpd.points_from_xy(xs, ys), crs=gpd.read_file(dem.path).crs if hasattr(dem, "path") else None)

# 6) Recompute drainage basins *from those seeds only*
#    (tt3 typically supports passing seed points)
D = fd.drainagebasins(seeds=seeds_gdf)
labels = D.z.astype(np.int32)

# 7) Visualize (you should see fewer, larger exterior basins)
plt.figure()
plt.imshow(labels, interpolation="nearest", cmap="Pastel1")
plt.title(f"Exterior basins seeded by acc ≥ {MIN_AREA_CELLS} cells")
plt.axis("off"); plt.tight_layout(); plt.show()

#%%
import topotoolbox as ttb

# 1. Read your DEM into a GridObject
dem = ttb.read_tif("fullextent2.tif")   # or ttb.load_dem("...") if using a bundled example

# 2. Apply a 3x3 mean filter (QGIS 'Average 9' equivalent)
dem_smooth = dem.filter(method="mean", kernelsize=3)

# 3. Write out the smoothed DEM
ttb.write_tif(dem_smooth, "smoothed_dem_avg9.tif")
