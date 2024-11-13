import matplotlib.pyplot as plt

def show_images(titles, images, rows, cols):
    plt.figure(figsize=(15, 10))

    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
