import numpy as np


# Target-function parameters
A = np.array([[ 1.41741415,  1.99802315,  1.2253319 ],
              [-0.78822617, -0.61596058, -0.0047727 ],
              [ 0.84335428,  0.14448994,  0.51675322]])
B = np.array([-0.39255089, -0.83444712, -0.17785561])
C = np.array([-1.07769796, -0.03977989,  2.2432707 ])


def target_func(x):
    """Target function that neural networks aim to approximate.
    """
    y = np.sin(x.dot(A) + B) * C

    return y


def generate():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sample", type=int, default=500, help="Number of samples.")
    parser.add_argument("--out", default="dataset/train.csv", help="Output directory.")
    parser.add_argument("--noplot", dest="plot", action="store_false")
    args = parser.parse_args()

    X = []
    Y = []

    for _ in range(args.n_sample):
        # Sampling random inputs and corresponding outputs
        x = np.random.uniform(-2, 2, size=(3,))
        y = target_func(x)

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    dat = np.concatenate((X, Y), axis=1)

    # Save the input/output samples as a csv file
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savetxt(args.out, dat, delimiter=",")

    # Visualize the dataset if wanted
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d

        fig1 = plt.figure()
        ax1  = fig1.add_subplot(1, 1, 1, projection="3d")
        fig2 = plt.figure()
        ax2_1 = fig2.add_subplot(3, 1, 1)
        ax2_2 = fig2.add_subplot(3, 1, 2)
        ax2_3 = fig2.add_subplot(3, 1, 3)

        ax1.scatter(X[:, 0], X[:, 1], X[:, 2])
        ax2_1.scatter(X[:, 0], Y[:, 0])
        ax2_2.scatter(X[:, 1], Y[:, 1])
        ax2_3.scatter(X[:, 2], Y[:, 2])

        fig1.savefig(os.path.splitext(args.out)[0] + ".png")
        plt.show()


if __name__ == "__main__":
    generate()

