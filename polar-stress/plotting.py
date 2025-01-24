import matplotlib.pyplot as plt


def plot_DoLP_AoLP(DoLP, AoLP, filename="output.png"):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(DoLP, cmap="hot")
    plt.colorbar()
    plt.title("DoLP")
    plt.subplot(1, 2, 2)
    plt.imshow(AoLP, cmap="hsv")
    plt.colorbar()
    plt.title("AoLP")
    plt.savefig(filename)


def plot_fringe_pattern(intensity, isoclinic, filename="output.png"):
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(intensity, cmap="hot")
    plt.colorbar()
    plt.title("Fringe Pattern")

    plt.subplot(1, 2, 2)
    plt.imshow(isoclinic, cmap="hsv")
    plt.colorbar()
    plt.title("Isoclinic Angle")

    plt.savefig(filename)
