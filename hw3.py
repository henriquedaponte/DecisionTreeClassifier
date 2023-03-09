# Starter code for CS 165B HW3
# cd Documents/UCSB/4_Senior_Year/14_Winter2023/CMPSC_165B/Hw/Hw3

from sklearn.tree import DecisionTreeClassifier



def run_train_test(training_file, testing_file):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition. 

    Inputs:
        training_file: file object returned by open('training.txt', 'r')
        testing_file: file object returned by open('test1/2/3.txt', 'r')

    Output:
        Dictionary of result values 

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
        Example:
            return {
    			"gini":{
    				'True positives':0, 
    				'True negatives':0, 
    				'False positives':0, 
    				'False negatives':0, 
    				'Error rate':0.00
    				},
    			"entropy":{
    				'True positives':0, 
    				'True negatives':0, 
    				'False positives':0, 
    				'False negatives':0, 
    				'Error rate':0.00}
    				}
    """


    # Preparing training data
    training_file.readline()
    temp_train = [[int(y) for y in x.strip().split(" ")] for x in training_file]
    training_data = [[row[i] for i in range(len(row)) if i != 0] for row in temp_train]
    x_train = [row[:-1] for row in training_data]
    y_train = [row[-1] for row in training_data]

    # Preparing testing data
    testing_file.readline()
    temp_test = [[int(y) for y in x.strip().split(" ")] for x in testing_file]
    testing_data = [[row[i] for i in range(len(row)) if i != 0] for row in temp_test]
    x_test = [row[:-1] for row in testing_data]
    y_test = [row[-1] for row in testing_data]


    # Initializing Gini performance metrics
    gTP = 0
    gTN = 0
    gFP = 0
    gFN = 0
    gERR = 0.00

    # Initializing Entropy performance metrics
    eTP = 0
    eTN = 0
    eFP = 0
    eFN = 0
    eERR = 0.00

    
    # Creating Decision Tree Classifier based on Gini impurity measure
    gini = DecisionTreeClassifier( criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=4, random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)
    gini.fit(x_train, y_train) # Training the data for Gini
    gPred = (gini.predict(x_test)).tolist() # Performing classification for Gini

    # Evaluating performance for Gini
    for i, j in zip(gPred, y_test):
        if(i == 1 and j == 1):
            gTP += 1
        if(i == 1 and j == 0):
            gFP += 1
        if(i == 0 and j == 1):
            gFN += 1
        if(j == 0 and i == 0):
            gTN += 1

    gERR = (gFP + gFN) / (gTP + gTN + gFP + gFN) # Calculating error rate for Gini


    # Creating Decision Tree Classifier based on Entropy impurity measure
    entropy = DecisionTreeClassifier( criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=4, random_state=0, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)
    entropy.fit(x_train, y_train)
    ePred = (entropy.predict(x_test)).tolist() # Performing classification for Entropy

    # Evaluating performance for Entropy
    for i, j in zip(ePred, y_test):
        if(i == 1 and j == 1):
            eTP += 1
        if(i == 1 and j == 0):
            eFP += 1
        if(i == 0 and j == 1):
            eFN += 1
        if(j == 0 and i == 0):
            eTN += 1

    eERR = (eFP + eFN) / (eTP + eTN + eFP + eFN) # Calculating Error Rate for Entropy


    # Printing results
    print("gini:", '\n')
    print("True positives: ", gTP, '\n')
    print("True negatives: ", gTN, '\n')
    print("False positives: ", gFP, '\n')
    print("False negatives: ", gFN, '\n')
    print("Error rate: ", gERR, '\n')
    print('\n')
    print("entropy:", '\n')
    print("True positives: ", eTP, '\n')
    print("True negatives: ", eTN, '\n')
    print("False positives: ", eFP, '\n')
    print("False negatives: ", eFN, '\n')
    print("Error rate: ", eERR, '\n')

    # Returning Results
    return {
    			"gini":{
    				'True positives':gTP, 
    				'True negatives':gTN, 
    				'False positives':gFP, 
    				'False negatives':gFN, 
    				'Error rate':gERR
    				},
    			"entropy":{
    				'True positives':eTP, 
    				'True negatives':eTN, 
    				'False positives':eFP, 
    				'False negatives':eFN, 
    				'Error rate':eERR}
    				}


    pass


#######
# The following functions are provided for you to test your classifier.
#######

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw3.py [training file path] [testing file path]
    """
    import sys

    training_file = open(sys.argv[1], "r")
    testing_file = open(sys.argv[2], "r")

    run_train_test(training_file, testing_file)

    training_file.close()
    testing_file.close()

