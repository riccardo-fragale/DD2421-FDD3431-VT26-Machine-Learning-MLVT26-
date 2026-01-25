import monkdata as m
import dtree as dt

def main():

    monk_train = [m.monk1, m.monk2, m.monk3]

    monk_test = [m.monk1test, m.monk2test, m.monk3test]
    
    print('********** Assignment 1 **********')
    # entropy
    for i in range(3):
        print(f"entropy MONK{i+1} = {dt.entropy(monk_train[i])}")
   


    print('********** Assignment 3 **********')
    # information gain first decision in MONK 1
    for attribute in m.attributes:
        print(f"database monk1, information gain of the attribute {attribute} = {dt.averageGain(monk_train[0],attribute)}")
        print(f"database monk2, information gain of the attribute {attribute} = {dt.averageGain(monk_train[1],attribute)}")
        print(f"database monk3, information gain of the attribute {attribute} = {dt.averageGain(monk_train[2],attribute)}")
    
    print('********** Assignment 5 **********')
    for i in range(3):
        d_tree = dt.buildTree(monk_train[i],m.attributes)
        print(f"Dataset MONK{i+1}: Train error: {1 - dt.check(d_tree,monk_train[i])}")
        print(f"Dataset MONK{i+1}: Test error: {1 - dt.check(d_tree,monk_test[i])}")





if __name__ == '__main__':
    main()


