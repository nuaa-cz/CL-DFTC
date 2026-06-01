import sys
sys.path.append('..')
import numpy as np
import math

from config import Config
from utils import tool_funcs
from utils.rdp import rdp
from utils.cellspace import CellSpace
from utils.tool_funcs import truncated_rand


def raw(src):
    return src


def simplify(src):
    # src: [[lon, lat], [lon, lat], ...]
    return rdp(src, epsilon = Config.traj_simp_dist)


def scale_factor(n, i):

    if i == n - 1:
        return 0

    return np.exp(-i / Config.traj_shift_decay_factor) * Config.traj_shift_max_offset


def shift(src):
    simplify_src = simplify(src)
    n = len(simplify_src)

    return [
        [p[0] + truncated_rand(mu=0, sigma=0.5, factor=scale_factor(n, i)),
         p[1] + truncated_rand(mu=0, sigma=0.5, factor=scale_factor(n, i))]
        for i, p in enumerate(simplify_src)
    ]


def mask(src):
    l = len(src)
    arr = np.array(src)
    mask_idx = np.random.choice(l, int(l * Config.traj_mask_ratio), replace = False)
    return np.delete(arr, mask_idx, 0).tolist()



def get_aug_fn(name: str):
    return {'raw': raw, 'simplify': simplify, 'shift': shift, 'mask': mask}.get(name, None)


# pair-wise conversion -- structural features and spatial feasures
def merc2cell2(src, cs: CellSpace):
    # convert and remove consecutive duplicates

    tgt = [ (cs.get_cellid_by_point(*p), p) for p in src]
    tgt = [v for i, v in enumerate(tgt) if i == 0 or v[0] != tgt[i-1][0]]
    tgt, tgt_p = zip(*tgt)
    return tgt, tgt_p


def generate_spatial_features(src, cs: CellSpace):
    # src = [length, 2]
    tgt = []
    lens = []
    for p1, p2 in tool_funcs.pairwise(src):
        lens.append(tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1]))

    for i in range(1, len(src) - 1):
        dist = (lens[i-1] + lens[i]) / 2
        dist = dist / (Config.trajcl_local_mask_sidelen / 1.414)  # float_ceil(sqrt(2))

        radian = math.pi - math.atan2(src[i-1][0] - src[i][0],  src[i-1][1] - src[i][1]) \
                        + math.atan2(src[i+1][0] - src[i][0],  src[i+1][1] - src[i][1])
        radian = 1 - abs(radian) / math.pi

        x = (src[i][0] - cs.x_min) / (cs.x_max - cs.x_min)
        y = (src[i][1] - cs.y_min)/ (cs.y_max - cs.y_min)
        tgt.append( [x, y, dist, radian] )

    x = (src[0][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[0][1] - cs.y_min)/ (cs.y_max - cs.y_min)
    tgt.insert(0, [x, y, 0.0, 0.0] )
    
    x = (src[-1][0] - cs.x_min) / (cs.x_max - cs.x_min)
    y = (src[-1][1] - cs.y_min)/ (cs.y_max - cs.y_min)
    tgt.append( [x, y, 0.0, 0.0] )
    # tgt = [length, 4]
    return tgt


def generate_semantics_features(src):
    # src = [length, 2]
    tgt = []
    headings = []
    for i in range(1, len(src)):
        dx = src[i][0] - src[i - 1][0]
        dy = src[i][1] - src[i - 1][1]
        heading = math.atan2(dy, dx)
        headings.append(heading)

    headings.append(headings[-1])

    adjusted_headings = [headings[-1]]

    for i in range(len(headings) - 2, -1, -1):
        delta_heading = headings[i] - headings[i + 1]
        if delta_heading > math.pi:
            delta_heading -= 2 * math.pi
        elif delta_heading < -math.pi:
            delta_heading += 2 * math.pi
        adjusted_heading = adjusted_headings[-1] + delta_heading
        adjusted_headings.append(adjusted_heading)

    adjusted_headings.reverse()

    total_length = 0
    distances = [0]
    for i in range(1, len(src)):
        dx = src[i][0] - src[i - 1][0]
        dy = src[i][1] - src[i - 1][1]
        dist = math.sqrt(dx ** 2 + dy ** 2)
        total_length += dist
        distances.append(total_length)

    remaining_distances = [total_length - dist for dist in distances]

    for i in range(len(src)):
        # x, y = src[i]
        adjusted_heading = adjusted_headings[i]
        remaining_distance = remaining_distances[i]
        tgt.append([adjusted_heading, remaining_distance])
        # tgt.append([x, y, adjusted_heading, remaining_distance])

    # tgt = [length, 2]
    return tgt


def traj_len(src):
    length = 0.0
    for p1, p2 in tool_funcs.pairwise(src):
        length += tool_funcs.l2_distance(p1[0], p1[1], p2[0], p2[1])
    return length

