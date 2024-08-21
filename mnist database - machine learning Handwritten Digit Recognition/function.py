import matplotlib.pyplot as plt


def sample_results(images_test_0, pred, k=0):
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            image = images_test_0.iloc[k].values.reshape(28, 28)
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Prediction: {pred[k]}", fontsize=8)
            ax.title.set_size(10)

            k = k + 1
            ax.axis("off")

    plt.show()
    return