import csv

def save_acc_loss(acc, loss, filename):
    with open(filename+'.csv', 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['epoch', 'acc', 'loss'])
        outcomes = zip(acc, loss)
        
        for idx, item in enumerate(outcomes):
            writer.writerow([idx, item[0], item[1]])

def save_maxlocals(maxlocals, filename):
    with open(filename+'.csv', 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['idx','ang', 'correlation', 'shape','h', 'w', 'x', 'y'])
        
        for idx, item in enumerate(maxlocals):
            writer.writerow([idx, item.ang, item.correlation, item.shape, item.h, item.w, item.x, item.y])