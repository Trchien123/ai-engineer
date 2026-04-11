import os
import matplotlib.pyplot as plt
import math
import numpy as np


def make_txt(path,txt_name):
    for file in os.listdir(path):
        if file.endswith('.txt'):
            with open(path+file, 'r') as f:
                for line in f:
                    with open(txt_name, 'a') as f:
                        f.write(line)
                    f.close()


def plot_image(txt_name, lamda):
    weight=[]
    height=[]
    data = []
    with open(txt_name, 'r') as f:
        for line in f:
            b1 = float(line.split(' ')[3])
            b2 = float(line.split(' ')[4])
            height.append(b1*640)
            weight.append((b2*640))
            data.append(b1*640)
            data.append(b2*640)

        test = np.vstack((height, weight)).T  # Combine into a data array
        size_test = np.prod(test, axis=1)
        largest_2_percent_threshold = np.percentile(size_test, 98)
        smallest_5_percent_threshold = np.percentile(size_test, 5)

        colors = ['orange' if size >= largest_2_percent_threshold else
                  'green' if size <= smallest_5_percent_threshold else
                  'pink' for size in size_test]

        height.sort()
        weight.sort()

        weight_min = 0
        weight_max = 0
        height_min = 0
        height_max = 0

        len_min = int(len(height)*0.05)
        len_max = int(len(height)*0.98)

        for i in range(len_min):
            weight_min += weight[i]
            height_min += height[i]
        weight_min = weight_min / len_min
        height_min = height_min / len_min

        for i in range(len_max, int(len(height))):
            weight_max += weight[i]
            height_max += height[i]
        weight_max = weight_max / (len(height) - len_max)
        height_max = height_max / (len(height) - len_max)

        anchor_min = math.ceil(min(weight_min,height_min))
        anchor_max = math.ceil(max(weight_max,height_max))
        anchor_avg = math.ceil(sum(data)/len(data))
        print('anchor_min：', anchor_min)
        print('anchor_avg：', anchor_avg)
        print('anchor_max：', anchor_max)

        anchor_P5 = anchor_max
        anchor_P1 = anchor_min
        anchor_P3 = anchor_avg
        anchor_P2 = (anchor_P1 + anchor_P3) / 2
        anchor_P4 = (anchor_P3 + anchor_P5) / 2
        print('anchor_P1:', anchor_P1)
        print('anchor_P2:', anchor_P2)
        print('anchor_P3:', anchor_P3)
        print('anchor_P4:', anchor_P4)
        print('anchor_P5:', anchor_P5)

        n1 = max(math.ceil((1.0*lamda*anchor_P1-3)/8), 1)
        n2 = max(math.ceil((1.0*lamda*anchor_P2 - 7 - 8*n1)/16), 1)
        n3 = max(math.ceil((1.0*lamda*anchor_P3 - 15 - 8*n1 - 16*n2) / 32), 1)
        n4 = max(math.ceil((1.0*lamda*anchor_P4 - 31 - 8*n1 - 16*n2 - 32*n3) / 64), 1)
        n5 = max(math.ceil((1.0*lamda*anchor_P5 - 63 - 8*n1 - 16*n2 - 32*n3 - 64*n4) / 128), 1)

        print('n1:', n1)
        print('n2:', n2)
        print('n3:', n3)
        print('n4:', n4)
        print('n5:', n5)

        plt.scatter(test[:, 1], test[:, 0], marker='.', s=10, alpha=0.6, c=colors)
        plt.scatter(math.ceil(sum(height)/len(height)), math.ceil(sum(weight)/len(weight)), marker='.', s=100, alpha=1, c='red')

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.title('TT100K - Train set')
        plt.xlabel("Weight")
        plt.ylabel("Height")
        plt.xlim((0, 126))
        plt.ylim((0, 105))

        # Create custom legends
        legend_labels = ['2% Largest', '5% Smallest', 'Mean Value', 'Others']
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[0],
                       markerfacecolor='orange', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[1],
                       markerfacecolor='green', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[2],
                       markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[3],
                       markerfacecolor='pink', markersize=10)
        ]
        plt.legend(handles=handles, loc='lower right', title='Traffic signs')

        plt.savefig('dataset.png',dpi=1500)
        plt.show()


if __name__ == '__main__':
    path = '../data/damaged-sign3/final_dataset_split_5/train/labels/'
    txt_name = './aus_dataset_combined.txt'
    if not os.path.exists(txt_name):
        make_txt(path,txt_name)
    plot_image(txt_name, lamda=4)