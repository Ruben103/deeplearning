import matplotlib.pyplot as plt

def multi(data):
    f = plt.figure()
    f.cmap = 'gray'
    for i in range(25):
        f.add_subplot(5,5,i+1)
        plt.imshow(data[i+25])
        plt.axis("off")
        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=1, bottom=0, left=0,
                            right=1)
    plt.show()
