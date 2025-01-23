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
