import sleap
import numpy as np


def clean_sleap_file(slp_file, num_animals, dist_thresh=50, as_h5=False, filter_score=False):
    labels = sleap.load_file(slp_file)
    og_track_list = []
    track_cents = []
    for n in range(num_animals):
        og_track_list.append(labels.tracks[n])
        if n >= len(labels[0].instances):
            track_cents.append(np.array([0,0]))
        else:
            track_cents.append(labels[0].instances[n].centroid)
    c = 1
    for l in labels.labeled_frames:
        if c % 1000 == 0:
            print('Cleaning slp file, on frame: ', c)
        out_ins = []
        insts = l.instances
        if filter_score:
            insts = filter_by_score(insts)
        num_i = len(insts)
        if num_i > 0:
            track_dists = np.zeros((num_animals, num_i))
            for ind, z in enumerate(zip(og_track_list, track_cents)):
                t, tc = z[:]
                for ind2, i in enumerate(insts):
                    d = np.linalg.norm(tc - i.centroid)
                    track_dists[ind, ind2] = d
            closest_i = np.argmin(track_dists, axis=1)
            for c_i in closest_i:
                dists = track_dists[:, c_i]
                min_t = np.argmin(dists)
                min_dist = dists[min_t]
                if min_dist < dist_thresh:
                    insts[c_i].track = og_track_list[min_t]
                    out_ins.append(insts[c_i])
            l.instances = out_ins
        c += 1
    labels.tracks[num_animals+1:] = []
    out_slp_file = slp_file.split('.')[0]
    if as_h5:
        out_slp_file += '_fixed.h5'
        labels.export(out_slp_file)
    else:
        out_slp_file += '_fixed.slp'
        labels.save(out_slp_file)
    return out_slp_file


def slp_to_h5(slp_file):
    labels = sleap.load_file(slp_file)
    out_slp_file = slp_file.split('.')[0] + '.h5'
    labels.export(out_slp_file)

def filter_by_score_save(slp_file, score_thresh=0.7):
    labels = sleap.load_file(slp_file)
    c = 1
    for l in labels.labeled_frames:
        if c % 1000 == 0:
            print('Cleaning slp file, on frame: ', c)
        out_ins = []
        num_i = len(l.instances)
        if num_i > 0:
            insts = l.instances
            for i in insts:
                if i.score > score_thresh:
                    out_ins.append(i)
        l.instances = out_ins
        c += 1
    out_slp_file = slp_file.split('.')[0] + '_fixed.h5'
    labels.export(out_slp_file)
    return out_slp_file

def filter_by_score(slp_insts, score_thresh=0.7):
    num_i = len(slp_insts)
    out_ins = []
    if num_i > 0:
        for i in slp_insts:
            if i.score > score_thresh:
                out_ins.append(i)
    return out_ins
