import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

os.chdir(sys.path[0])

algs_fmnist_iid=[
    800,
    [65,75,85],
    'FMNIST_iid_1005_1505',
    ('FedPer','ASync_10_0.3'),
    ('FedAsync', 'AFO_13_1.0'),
    ('FedAvg+Topk', 'Sync_20_0.5'),
    ('FedAvg', 'Sync_20_1.0'),
    ('FedLuck','ASync_auto_1_1'),
    # ('FedBuff','FedBuff_20_0.5'),
]

algs_fmnist_niid=[
    800,
    [65,75,85],
    'FMNIST_non_iid_1004_1911_good',
    ('FedPer','ASync_10_0.3'),
    ('FedAsync', 'AFO_13_1.0'),
    ('FedAvg+Topk', 'Sync_20_0.5'),
    ('FedAvg', 'Sync_20_1.0'),
    ('FedLuck','ASync_auto_1_1'),
]

# algs_cifar10_iid=[
#     6000,
#     [50,60,70],
#     'CIFAR10_iid_1211_1152',
#     ('FedPer','../../results_Dec/CIFAR10_iid_1210_1329/ASync_8_0.05'),
#     ('FedAsync', '../../results_Dec/CIFAR10_iid_1210_1618/AFO_20_1.0'),
#     ('FedAvg+Topk', 'Sync_20_0.2'),
#     ('FedAvg', 'Sync_20_1.0'),
#     ('FedLuck','../../results_Dec/CIFAR10_iid_1210_1114/ASync_auto_1_1'),
#     # ('FedBuff','FedBuff_15_0.05'),
# ]

algs_cifar10_iid=[
    6000,
    [50,60,70],
    'CIFAR10_iid_1213_2132',
    ('FedPer','../CIFAR10_iid_1213_1746/ASync_8_0.1'),
    ('FedAsync', '../CIFAR10_iid_1130_1512/AFO_36_1.0'),
    ('FedAvg+Topk', 'Sync_13_0.2'),
    ('FedAvg', '../CIFAR10_iid_1130_1512/Sync_20_1.0'),
    ('FedLuck','ASync_auto_1_0.01'),
]

algs_cifar10_niid=[
    6000,
    [50,60,70],
    'CIFAR10_non_iid_1210_1944',
    ('FedPer','ASync_8_0.05'),
    ('FedAsync', 'AFO_20_1.0'),
    ('FedAvg+Topk', 'Sync_13_0.2'),
    ('FedAvg', 'Sync_13_1.0'),
    ('FedLuck','ASync_auto_1_1'),
]

algs_cifar100_iid=[
    6000,
    [50,60,70],
    'CIFAR100_iid_1213_2132',
    ('FedPer','../CIFAR100_iid_1213_1746/ASync_8_0.1'),
    ('FedAsync', '../CIFAR100_iid_1130_1512/AFO_36_1.0'),
    ('FedAvg+Topk', 'Sync_13_0.2'),
    ('FedAvg', '../CIFAR100_iid_1130_1512/Sync_20_1.0'),
    ('FedLuck','ASync_auto_1_0.01'),
]

algs_CIFAR100_niid=[
    6000,
    [50,60,70],
    'CIFAR100_non_iid_1210_1944',
    ('FedPer','../CIFAR100/ASync_8_0.05'),
    ('FedAsync', '../CIFAR100/AFO_20_1.0'),
    ('FedAvg+Topk', '../CIFAR100/Sync_13_0.2'),
    ('FedAvg', '../CIFAR100/Sync_13_1.0'),
    ('FedLuck','../CIFAR100/ASync_auto_1_1'),
]

algs_sc_iid=[
    6000,
    [50,56,62],
    'SC_iid_1212_2214',
    ('FedPer','ASync_5_0.16'),
    ('FedAsync', '../SC_iid_1212_1252/AFO_5_1.0'),
    ('FedAvg+Topk', 'Sync_8_0.5'),
    ('FedAvg', 'Sync_8_1.0'),
    ('FedLuck','../SC_iid_1212_1049/ASync_auto_1_1'),
]

algs_sc_niid=[
    6000,
    [50,60,63],
    'SC_niid',
    ('FedPer','ASync_20_0.16 copy'),
    ('FedAsync', 'AFO_13_1.0'),
    ('FedAvg+Topk', 'Sync_20_0.5'),
    ('FedAvg', 'Sync_20_1.0'),
    ('FedLuck','ASync_auto_1_1'),
]



def read_data(algs):
    algs_data = []
    afo_name='FedAsync'
    # assert afo_name in [name for name, _ in algs], "FedAsync is required"
    for name, path in algs:
        with open(os.path.join(PLOT_ROOT, path, 'time.txt'), 'r') as time_file, \
             open(os.path.join(PLOT_ROOT, path, 'global_acc.txt'), 'r') as acc_file, \
             open(os.path.join(PLOT_ROOT, path, 'communication_cost.txt'), 'r') as cost_file:

            time_lines = [float(line.strip()) for line in time_file.readlines()]
            acc_lines = [100*float(line.strip()) for line in acc_file.readlines()]
            cost_lines = [float(line.strip()) for line in cost_file.readlines()]

            start_time = time_lines[0]
            time_lines = [time - start_time for time in time_lines]

            time_lines = [0] + time_lines
            acc_lines = [0] + acc_lines
            cost_lines = [0] + cost_lines

            for i in range(len(acc_lines)):
                test_step = 10 if name==afo_name else 1
                time_index = i * test_step
                if time_index < len(time_lines):
                    algs_data.append((name, time_lines[time_index], acc_lines[i], cost_lines[time_index]))

    return pd.DataFrame(algs_data, columns=['Algorithm', 'Time', 'Accuracy', 'Cost'])

def plot_time_accuracy(raw_data, clip_time):
    # clip time
    data=raw_data[raw_data['Time']<=clip_time]
    palette = [(125,183,138), (104,103,137), (73,117,154),  (180,116,107),(180,33,100), (53,92,135)]
    palette = [(r/255, g/255, b/255) for r, g, b in palette]
    colors = {alg[0]: color for alg, color in zip(algs, palette)}

    mpl.rcParams['font.size'] = 20
    plt.figure(figsize=(8,6))
    # different dash level for different algorithms
    # sns.lineplot(data=data, x='Time', y='Accuracy', hue='Algorithm',palette=colors, style='Algorithm', linewidth=2.5, dashes=False)
    sns.lineplot(data=data, x='Time', y='Accuracy', hue='Algorithm',palette=colors, style='Algorithm', linewidth=2.5, dashes=False)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    fedlg_index = labels.index('FedLuck')
    handles.insert(0, handles.pop(fedlg_index))
    labels.insert(0, labels.pop(fedlg_index))
    
    
    ax.legend_.remove()
    plt.legend(handles, labels, title=None)

    # Adjust line width in legend
    for legend_handle in ax.legend_.legendHandles:
        legend_handle.set_linewidth(2.5)

    # Add horizontal lines
    for i in range(1, 11):
        plt.axhline(y=10 * i, color='gray', linestyle='--', linewidth=0.6)

    plt.title('')
    plt.ylim(0, data['Accuracy'].max())
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Test Accuracy (%)')
    #save as pdf
    plt.savefig(os.path.join(PLOT_ROOT,'time_acc.pdf'),bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_ROOT,'time_acc.png'),bbox_inches='tight')
    # plt.show()

def plot_accuracy_cost(data, target_accuracies):
    # Create DataFrame for plotting
    plot_data = []
    for target_acc in target_accuracies:
        print("acc={}%:".format(target_acc))
        for alg in data['Algorithm'].unique():
            subset = data[data['Algorithm'] == alg]
            costs = subset[subset['Accuracy'] >= target_acc]['Cost']
            times = subset[subset['Accuracy'] >= target_acc]['Time'].iloc[0]
            if not costs.empty:
                plot_data.append((alg, target_acc,times ,costs.min()/1024)) # multiply by 100 to match accuracy scale
            print(f'{alg}: {times:.2f} s, {costs.min()/1024:.2f} GB')
    plot_data = pd.DataFrame(plot_data, columns=['Algorithm', 'Target Accuracy', 'Time' , 'Cost'])
    # Reorder Algorithms with 'FedLG' first
    order_list=['FedLuck', 'FedPer', 'FedAsync','FedAvg+Topk','FedAvg', 'FedBuff' ]
    plot_data['Algorithm'] = plot_data['Algorithm'].astype(pd.CategoricalDtype(categories=order_list, ordered=True))
    plot_data = plot_data.sort_values('Algorithm')
    # Color palette
    palette = [
        (181, 211, 217),
        (111, 160, 172),
        (93, 136, 123),
        (252, 238, 226),
        (44, 65, 80),
        (111, 211, 172),
    ]
    palette = [(r/255, g/255, b/255) for r, g, b in palette]
    colors = {alg[0]: color for alg, color in zip(algs, palette)}

    # Plot
    mpl.rcParams['font.size'] = 20
    plt.figure(figsize=(8,6))
    sns.barplot(data=plot_data, x='Target Accuracy', y='Cost', hue='Algorithm', palette=colors, edgecolor='black')
    plt.ylabel('Communication Consumption (GB)')
    plt.xlabel('Target Accuracy (%)')

    # Determine the range and interval for y-axis labels
    max_cost = plot_data['Cost'].max()
    if(max_cost<10):
        n_slice=2
    else:
        n_slice = 5
    interval = int(max_cost / n_slice)
    
    # Add horizontal lines and set y-tick labels
    y_ticks = []
    for i in range(1, n_slice+2):
        y_value = interval * i
        plt.axhline(y=y_value, color='gray', linestyle='--', linewidth=0.6)
        y_ticks.append(y_value)
    
    plt.yticks(y_ticks) # Set the y-tick labels

    # Adjust legends
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend_.remove()
    plt.legend(handles, labels, title=None)
    plt.legend(loc='upper left')

    plt.savefig(os.path.join(PLOT_ROOT,'acc_cost.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_ROOT,'acc_cost.png'), bbox_inches='tight')
    # plt.show()

def plot(plot_settings):
    clip_time=plot_settings[0]
    target_accuracies = plot_settings[1] #[50, 60, 70] # target acc
    global PLOT_ROOT, algs
    PLOT_ROOT=os.path.join('results_Dec',plot_settings[2])
    algs=plot_settings[3:]
    # Read the data
    data = read_data(algs)
    
    # plot
    # plot_accuracy_cost(data, target_accuracies)
    plot_time_accuracy(data, clip_time)

if __name__ == '__main__':
    plot(algs_cifar10_iid)
    # plot(algs_fmnist_iid)
    # plot(algs_sc_iid)