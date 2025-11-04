import numpy as np

def equirect_xy_to_lonlat(x, y, W, H):
    """Convert pixel (x,y) to (lon,lat) in degrees for equirectangular projection.
    lon in [-180, 180], lat in [-90, 90].
    """
    lon = (x / W) * 360.0 - 180.0
    lat = 90.0 - (y / H) * 180.0
    return float(lon), float(lat)

def weighted_centroid(heatmap: np.ndarray):
    """Return centroid (x,y) of heatmap in pixel coords.
    heatmap expected in [0,1], shape (H,W).
    """
    H, W = heatmap.shape
    h = heatmap.astype(np.float64)
    s = h.sum()
    if s <= 1e-12:
        return W/2.0, H/2.0
    ys = np.arange(H).reshape(H,1)
    xs = np.arange(W).reshape(1,W)
    cx = float((h * xs).sum() / s)
    cy = float((h * ys).sum() / s)
    return cx, cy

def topk_peaks(heatmap: np.ndarray, k:int=1):
    """Return top-k peak (x,y,value) in pixel coords."""
    H, W = heatmap.shape
    flat = heatmap.reshape(-1)
    idx = np.argpartition(-flat, min(k, flat.size)-1)[:k]
    peaks = []
    for i in idx:
        y = i // W
        x = i % W
        peaks.append((float(x), float(y), float(heatmap[y, x])))
    peaks.sort(key=lambda t: -t[2])
    return peaks