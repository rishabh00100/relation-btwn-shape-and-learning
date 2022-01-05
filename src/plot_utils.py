import numpy as np
# ! pip install --update matplotlib
import matplotlib
import matplotlib.pyplot as plt

def plot_saliency_by_features(data_matrix, num_classes, num_features, fig_title='', fig_width=10, fig_height=40):
    classes = [i for i in range(num_classes)]
    features = [i for i in range(num_features)]

    fig, ax = plt.subplots()
    fig.set_size_inches(fig_width, fig_height)
    im = ax.imshow(data_matrix)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(classes)))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # for i in range(len(features)):
    #     for j in range(len(classes)):
    #         text = ax.text(j, i, avg_saliency[j][i],
    #                        ha="center", va="center", color="w")

    ax.set_title(fig_title)
    fig.tight_layout()
    plt.show()