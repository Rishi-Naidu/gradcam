# Placeholder for evaluation metrics (e.g., angular error between predicted FoV and ground truth gaze).
# You can implement:
# - Great-circle distance on the sphere (Haversine) between (lon,lat) pairs.
# - Hit rate within a threshold FoV radius.
import math

def angular_distance_deg(lon1, lat1, lon2, lat2):
    # convert to radians
    rlon1, rlat1 = math.radians(lon1), math.radians(lat1)
    rlon2, rlat2 = math.radians(lon2), math.radians(lat2)
    # spherical law of cosines
    cosang = (math.sin(rlat1)*math.sin(rlat2) + math.cos(rlat1)*math.cos(rlat2)*math.cos(rlon1-rlon2))
    cosang = min(1.0, max(-1.0, cosang))
    return math.degrees(math.acos(cosang))