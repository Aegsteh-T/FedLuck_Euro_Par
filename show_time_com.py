import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# need FedAvg
algs_cifar10=[
    ('FedPer','CIFAR10/baseline_FedPer/VGG11s_3_CIFAR10_20_0.2Period',True,1),
    ('FedAsync', 'CIFAR10/baseline_FedAsync/VGG11s_3_CIFAR10_1.0AFO', True, 10),
    ('FedAvg+Topk', 'CIFAR10/baseline_FedAvg_topk/VGG11s_3_CIFAR10_20_FedAvg', False, 1),
    ('FedAvg', 'CIFAR10/baseline_FedAvg/VGG11s_3_CIFAR10_20_FedAvg', False, 1),
    # ('FedLG','CIFAR10/baseline_FedLG/VGG11s_3_CIFAR10_20_0.2Period',True,1),
    ('FedLuck','CIFAR10/baseline_FedLG2/VGG11s_3_CIFAR10_20_0.2Period',True,1),
]

algs_cifar100=[
    ('FedPer','CIFAR100/baseline_FedPer/ResNet18_CIFAR100_20_0.2Period',True,1),
    ('FedAsync', 'CIFAR100/baseline_FedAsync/ResNet18_CIFAR100_1.0AFO', True, 10),
    ('FedAvg+Topk', 'CIFAR100/baseline_FedAvg_topk/ResNet18_CIFAR100_20_FedAvg', False, 1),
    ('FedAvg', 'CIFAR100/baseline_FedAvg/ResNet18_CIFAR100_20_FedAvg', False, 1),
    ('FedLuck','CIFAR100/baseline_FedLG2/ResNet18_CIFAR100_20_0.2Period',True,1),
]

algs_fmnist=[
    ('FedPer','FMNIST/baseline_FedPer/CNN1_FMNIST_20_0.1Period',True,1),
    ('FedAsync', 'FMNIST/baseline_FedAsync/CNN1_FMNIST_1.0AFO', True, 10),
    ('FedAvg+Topk', 'FMNIST/baseline_FedAvg_topk/CNN1_FMNIST_20_FedAvg', False, 1),
    ('FedAvg', 'FMNIST/baseline_FedAvg/CNN1_FMNIST_20_FedAvg', False, 1),
    ('FedLuck','FMNIST/baseline_FedLG/CNN1_FMNIST_20_0.2Period',True,1),
]
algs_fmnist_niid=[
    ('FedPer','FMNIST_niid/baseline_FedPer/CNN1_FMNIST_20_0.1Period',True,1),
    ('FedAsync', 'FMNIST_niid/baseline_FedAsync/CNN1_FMNIST_1.0AFO', True, 10),
    ('FedAvg+Topk', 'FMNIST_niid/baseline_FedAvg_topk/CNN1_FMNIST_20_FedAvg', False, 1),
    ('FedAvg', 'FMNIST_niid/baseline_FedAvg/CNN1_FMNIST_20_FedAvg', False, 1),
    ('FedLuck','FMNIST_niid/baseline_FedLG/CNN1_FMNIST_20_0.2Period',True,1),
]

# need FedAvg, AFO
algs_cifar10_niid=[
    ('FedPer','CIFAR10_niid/baseline_FedPer/VGG11s_3_CIFAR10_20_0.2Period',True,1),
    ('FedAsync', 'CIFAR10_niid/baseline_FedAsync/VGG11s_3_CIFAR10_1.0AFO', True, 10),
    ('FedAvg+Topk', 'CIFAR10_niid/baseline_FedAvg_topk/VGG11s_3_CIFAR10_20_FedAvg', False, 1),
    ('FedAvg', 'CIFAR10_niid/baseline_FedAvg/VGG11s_3_CIFAR10_20_FedAvg', False, 1),
    ('FedLuck','CIFAR10_niid/baseline_FedLG/VGG11s_3_CIFAR10_20_0.2Period',True,1),
]


# METHOD='FedPer'
# ACC=0.65
algs=algs_cifar100

# clip=800
# clip=800


def read_data(algs):
    alg_data = {}

    for name, path, need_time_to_start, test_step in algs:
        # Initialize an empty list for each algorithm
        alg_data[name] = []

        # Open the time and accuracy files and read lines into float lists
        with open(os.path.join(path, 'time.txt'), 'r') as time_file, open(os.path.join(path, 'global_acc.txt'), 'r') as acc_file, open(os.path.join(path, 'communication_cost.txt'), 'r') as com_file:
            time_lines = [float(line.strip()) for line in time_file.readlines()]
            acc_lines = [float(line.strip()) for line in acc_file.readlines()]
            com_lines = [float(line.strip()) for line in com_file.readlines()]

            if need_time_to_start:
                # Subtract the start time from all time stamps
                start_time = time_lines[0]
                time_lines = [time - start_time for time in time_lines]
            
            time_lines=[0]+time_lines
            acc_lines=[0]+acc_lines
            com_lines=[0]+com_lines
            # Pair each time stamp with the corresponding accuracy value
            for i in range(len(acc_lines)):
                time_index = i * test_step
                com_index = i * test_step

                if time_index < len(time_lines):
                    alg_data[name].append((time_lines[time_index], acc_lines[i]))
            for i in range(len(acc_lines)):
                acc = acc_lines[i]
                if i * test_step < len(time_lines):
                    time = time_lines[i * test_step]
                    com = com_lines[i * test_step]
                if name == METHOD and acc >= ACC:
                    print('ACC:{}, METHOD:{}, time:{}, com:{}'.format(ACC, METHOD, round(time, 2), round(com/1024, 2)))
                    return
    return alg_data



def clip_data(alg_data, time_limit=1500):
    clipped_data = {}
    for name, data_points in alg_data.items():
        clipped_data[name] = [(time, acc) for time, acc in data_points if time <= time_limit]
    return clipped_data

def sparse_data(alg_data, names, sparse_step=3):
    sparse_data = {}
    for name, data_points in alg_data.items():
        if name in names:
            sparse_data[name]=[]
            idx=0
            for time, acc in data_points:
                idx+=1
                if idx % sparse_step ==0:
                    sparse_data[name].append((time,acc))
        else:
            sparse_data[name]=data_points
    return sparse_data
        


import pandas as pd

for acc in [0.65, 0.75,0.87]:
    ACC = acc
    for m in ['FedLuck', 'FedPer', 'FedAsync', 'FedAvg+Topk', 'FedAvg']:
        METHOD = m
        algs_data = read_data(algs)
    print('')


# algs_data=clip_data(algs_data, clip)
# data_frames = []
# for name, data_points in algs_data.items():
#     df = pd.DataFrame(data_points, columns=['Time', 'Accuracy'])
#     df['Algorithm'] = name
#     data_frames.append(df)

# data = pd.concat(data_frames, ignore_index=True)

# data['Accuracy_smooth'] = data.groupby('Algorithm')['Accuracy'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)

# data['Accuracy_smooth'] = data['Accuracy'] * 100
# no_marker_algs = ['FedLG', 'FedPer', 'FedAsync','FedAvg','FedAvg-no-compress']

# Separate the data into two dataframes
# df_no_marker = data[data['Algorithm'].isin(no_marker_algs)]
# df_with_marker = data[~data['Algorithm'].isin(no_marker_algs)]

# palette = [(145,173,158), (104,103,137), (73,117,154),  (180,116,107),(180,33,100)]
# palette = [(r/255, g/255, b/255) for r, g, b in palette]
# colors = {alg[0]: color for alg, color in zip(algs, palette)}



# # Change the font size
# mpl.rcParams['font.size'] = 20  # Change the size as per your requirement
# plt.figure(figsize=(8,6))

# # Create the plot for lines without markers
# lineplot=sns.lineplot(data=data, x='Time', y='Accuracy_smooth', hue='Algorithm',palette=colors, style='Algorithm', linewidth=2.5, dashes=False)

# ax = plt.gca()

# # Getting the current legend handles and labels
# handles, labels = ax.get_legend_handles_labels()

# # Rearranging the handles and labels so that 'FedLG' is at the top
# fedlg_index = labels.index('FedLuck')
# handles.insert(0, handles.pop(fedlg_index))
# labels.insert(0, labels.pop(fedlg_index))

# # Removing the existing legend
# ax.legend_.remove()

# # Adding the new legend with the rearranged handles and labels
# plt.legend(handles, labels, title=None)

# # Adjusting the line width in the legend
# for legend_handle in lineplot.legend_.legendHandles:
#     legend_handle.set_linewidth(2.5) # Adjust the linewidth as needed
# for i in range(1, 11):
#     plt.axhline(y=10 * i, color='gray', linestyle='--', linewidth=0.6)

# plt.title('')

# max_acc = data['Accuracy_smooth'].max()
# plt.ylim(0, max_acc)
# plt.xlabel('Elapsed Time (s)')
# plt.ylabel('Test Accuracy (%)')

# plt.savefig('./FMNIST_niid_time_acc.pdf',bbox_inches='tight')
# plt.savefig('./time-acc.png',bbox_inches='tight')
# plt.show()
