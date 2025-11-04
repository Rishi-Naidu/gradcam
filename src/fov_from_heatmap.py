import numpy as np
from .utils.spherical import weighted_centroid, equirect_xy_to_lonlat, topk_peaks

def fov_center_from_heatmap(hm: np.ndarray, k:int=1):
    """Return FoV center(s) from heatmap.
    If k==1 -> centroid; if k>1 -> top-k peaks.
    """
    H, W = hm.shape
    if k <= 1:
        cx, cy = weighted_centroid(hm)
        lon, lat = equirect_xy_to_lonlat(cx, cy, W, H)
        return [(cx, cy, lon, lat, float(hm[int(round(cy))%H, int(round(cx))%W]))]
    else:
        peaks = topk_peaks(hm, k)
        results = []
        for x,y,v in peaks:
            lon, lat = equirect_xy_to_lonlat(x, y, W, H)
            results.append((x, y, lon, lat, v))
        return results