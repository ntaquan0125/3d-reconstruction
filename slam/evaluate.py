import matplotlib.pyplot as plt
import numpy as np


def ominus(A, B):
    """
    Compute the relative 3D transformation between a and b.
    """
    return np.dot(np.linalg.inv(A), B)


def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    return np.linalg.norm(transform[0:3, 3])


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    return np.arccos(min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1) / 2)))


def distances_along_trajectory(traj):
    motion = [ominus(traj[i+1], traj[i]) for i in range(len(traj) - 1)]
    sum = 0
    for t in motion:
        sum += compute_distance(t)
    return sum


def calculate_trans_error(groundtruth, traj):
    diff = [ominus(groundtruth[i], traj[i]) for i in range(len(traj))]
    trans = []
    for t in diff:
        trans.append(compute_distance(t))
    return trans


def calculate_rot_error(groundtruth, traj):
    diff = [ominus(groundtruth[i], traj[i]) for i in range(len(traj))]
    rot = []
    for t in diff:
        rot.append(compute_angle(t))
    return rot


def load_groundtruth(filename, num_frames, skip_frames):
    rows, cols = 4, 4
    with open(filename) as f:
        groundtruth = []
        for i in range(0, num_frames):
            data = []
            f.readline()
            for _ in range(0, rows):
                data.append(list(map(float, f.readline().split()[:cols])))
            if i % skip_frames == 0:
                groundtruth.append(np.array(data))
    return groundtruth


if __name__ == '__main__':
    filename = 'data/traj.txt'
    groundtruth = load_groundtruth(filename, 2000, 1)
    x1 = [t[0, 3] for t in groundtruth]
    y1 = [t[1, 3] for t in groundtruth]
    plt.plot(x1, y1)
    plt.show()
