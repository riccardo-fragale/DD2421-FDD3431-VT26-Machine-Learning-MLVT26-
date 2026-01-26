import monkdata as m
import dtree as dt
import random
import numpy as np
import matplotlib.pyplot as plt
import statistics
from tabulate import tabulate



def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def get_best_tree(currTree):
    found_best_tree = False

    for newTree in dt.allPruned(currTree):
        if dt.check(newTree, monkval) > dt.check(currTree, monkval):
            found_best_tree = True
            currTree = newTree
    
    # Basic recursive part
    if found_best_tree:
        currTree = get_best_tree(currTree)  
    return currTree



if __name__ == '__main__':

    dataset_names = ('MONK-1','MONK-3')
    datasets_train = (m.monk1, m.monk3)
    datasets_test = (m.monk1test, m.monk3test)

    


    fractions = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    decimals = 3
    n = 500

    header = ['Dataset', 'Fraction', 'Mean error (n = {})'.format(n) , 'Standard deviation']

    for dataset_name, dataset_train, dataset_test in zip(dataset_names, datasets_train, datasets_test):

        data = []
        mean_errors = []
        std = []

        for fraction in fractions:

            errors = []

            for i in range(n):
                monktrain, monkval = partition(dataset_train, fraction)
                built_tree = dt.buildTree(monktrain,m.attributes)
                best_tree = get_best_tree(built_tree)
                errors.append(1 - dt.check(best_tree,dataset_test))
            
            
            mean_errors.append(round(statistics.mean(errors),decimals))

            std.append(round(statistics.stdev(errors),decimals))

            data.append([dataset_name, fraction, round(statistics.mean(errors),decimals), statistics.mean(std)])
        
        print(tabulate(data, header), '\n')
    
        plt.errorbar(fractions, mean_errors, yerr=std, marker='o')

        plt.title('{} (n = {})'.format('Monk Dataset', n))
        plt.xlabel('fraction')
        plt.ylabel('mean error')
        plt.legend(['Monk-1','Monk-3']) #loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3
        plt.show()


