import numpy as np
from skimage import io, color


def main():
    positives = io.imread_collection('data/positives/p*.png')
    negatives = io.imread_collection('data/negatives/n*.png')

    global u_pos, u_neg, cov, cov_inv, phi
    u_pos = np.mean([features(pos) for pos in positives], axis=0)
    u_neg = np.mean([features(neg) for neg in negatives], axis=0)

    cov = sum([np.outer(features(pos) - u_pos, features(pos) - u_pos)
               + np.outer(features(neg) - u_neg, features(neg) - u_neg)
               for pos, neg in zip(positives, negatives)]) / (len(positives) + len(negatives))
    cov_inv = np.linalg.inv(cov)

    phi = len(positives) / (len(positives) + len(negatives))

    # print('Covariance Matrix:', cov)
    print('Determinate:', np.linalg.det(cov))
    print('Definite positive: ', np.all(np.linalg.eigvals(cov) > 0))
    # print('Determinate:', np.linalg.det(cov))
    # print('Sqrt of Determinant:', np.sqrt(np.linalg.det(cov)))

    print('Highest y=0', prob_x_given_y(u_neg, 0))
    print('Highest x=0', prob_x_given_y(u_pos, 1))

    global numpy_cov
    numpy_cov = (np.cov([features(x) for x in positives], rowvar=False) + np.cov([features(x) for x in negatives],
                                                                                 rowvar=False)) / 2

    diff_mat = np.abs(np.abs(numpy_cov) - np.abs(cov))
    print('Max difference between Cov-Matrices', np.unravel_index(np.argmax(diff_mat, axis=None), diff_mat.shape))

    # print('Covariance Matrix:', numpy_cov)
    print('Determinate:', np.linalg.det(numpy_cov))
    print('Definite positive: ', np.all(np.linalg.eigvals(numpy_cov) > 0))
    # print('Sqrt of Determinant:', np.sqrt(np.linalg.det(numpy_cov)))

    print('POSITIVES:')
    for pos in positives:
        # feat = predict(features(pos))
        # print(feat)
        pass

    print('NEGATIVES:')
    for neg in negatives:
        # feat = predict(features(neg))
        # print(feat)
        pass


def prob_x_given_y(feat, y):
    term = 1 / ((2 * np.pi) ** (len(feat) / 2) * np.sqrt(np.linalg.det(cov)))
    if y == 0:
        return term * np.exp(-0.5 * np.dot(np.dot(feat - u_neg, cov_inv), feat - u_neg))
    elif y == 1:
        return term * np.exp(-0.5 * np.dot(np.dot(feat - u_pos, cov_inv), feat - u_pos))


def predict(feat):
    prob_x = prob_x_given_y(feat, 0) * prob_phi(0) + prob_x_given_y(feat, 1) * prob_phi(1)
    res = [(prob_x_given_y(feat, 0) * prob_phi(0)) / prob_x, (prob_x_given_y(feat, 1) * prob_phi(1)) / prob_x]
    return np.argmax(res), np.max(res)


def prob_phi(y):
    """
    Should always return 0.5
    """
    return phi ** y * (1 - phi) ** (1 - y)


def features(im):
    return feature(im[:, :, 0]) + feature(im[:, :, 1]) + feature(im[:, :, 2]) + feature(color.rgb2gray(im))
    # return feature(color.rgb2gray(im))


def feature(im):
    return [np.max(im), np.min(im), np.mean(im), np.std(im), np.var(im)]


if __name__ == '__main__':
    main()
