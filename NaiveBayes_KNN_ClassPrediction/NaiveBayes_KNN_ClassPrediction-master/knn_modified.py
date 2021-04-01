import math
import operator 

def handleTrainingDataset(filename, trainingSet=[] , testSet=[]):
    number=0
    with open("training.txt", "r") as File:  
        reader = [line.rstrip('\n') for line in File]
        for row in reader:
            number+=1
            if(number!=1):
                trainingSet.append(row)
            
    print("The trainingSet is:") 
    print(trainingSet)
    print("\n")
           
def handleTestDataset(filename, trainingSet=[] , testSet=[]):
    number=0
    with open("testing.txt", "r") as File:  
        reader = [line.rstrip('\n') for line in File]
        for row in reader:
            number+=1
            if(number!=1):
                testSet.append(row)
            
    print("The testingSet is:")
    print(testSet)
    print("\n")
    
#def euclideanDistance(instance1, instance2, length):
#    distance = 0
#    for x in range(length):
#        distance += pow((int(instance1[x]) - int(instance2[x])), 2)
#    return math.sqrt(distance)


def hamming_distance(instance1, instance2, length):
    distance = 0
    instance1=instance1.split()
    instance2=instance2.split()    
    for x in range(length):
        if(instance1[x]!=instance2[x]):
            distance +=1
    return (distance)

def KNearestNeighbors(trainingSet, testInstance, k):
    #distances = []
    distances2=[]
    length = len(testInstance.split())
    print("No of columns in TestSet:")
    print(length)
    print("\n")
    for x in range(len(trainingSet)):
        
        print("Taking the trainingSet:")
        print(x)
        print(testInstance)
        print(trainingSet[x])
        print("\n")
        dist=hamming_distance(testInstance, trainingSet[x], length)
        distances2.append((trainingSet[x], dist))
    distances2.sort(key=operator.itemgetter(1))
    print("dist2 using hamming dist is printed")
    print(distances2)
    
    neighbors = []
    
    for x in range(k):
        neighbors=distances2[x][0].split(" ")
        
        print("\n")
        print("The nearest neighbour to the testing dataset is:")
        print(neighbors)
        
        
        print("\n")
    return neighbors
	
def getKNN(neighbors):
    classVotes = {}
    
    for x in range(len(neighbors)):
        #neighbors=neighbors.split()
        output=neighbors[6]
        
        response = output
        
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0]

def mult():
    print("Multinomial Distribution:")
    import pandas as pd
    import numpy as np
    data = pd.read_csv("train.csv",header=None ,names= ['BodyTemperature','SkinCover','GivesBirth','HasLegs','Hibernate','Class'])
    data.head(8)
    obj_data = data.select_dtypes(include=['object']).copy()
    obj_data.head(8)
    cleanup_data = {"BodyTemperature": {"warm-blooded": 0, "cold-blooded": 1},"SkinCover": {"hair": 0, "scales": 1 , "none":2,"feathers":3},"GivesBirth": {"yes": 0, "no": 1},"HasLegs": {"yes": 0, "no": 1},"Hibernate": {"yes": 0, "no": 1},"Class": {"mammal":0, "reptile": 1, "fish":2,"amphibian":3,"bird":4}}
    obj_data.replace(cleanup_data, inplace=True)
    obj_data.head(8)
    data1 = pd.read_csv("test.csv",header=None ,names= ['BodyTemperature','SkinCover','GivesBirth','HasLegs','Hibernate'])
    data1.head(1)
    obj_data1 = data1.select_dtypes(include=['object']).copy()
    obj_data1.head(1)
    cleanup_data1 = {"BodyTemperature": {"warm-blooded": 0, "cold-blooded": 1},"SkinCover": {"hair": 0, "scales": 1 , "none":2,"feathers":3},"GivesBirth": {"yes": 0, "no": 1},"HasLegs": {"yes": 0, "no": 1},"Hibernate": {"yes": 0, "no": 1}}
    obj_data1.replace(cleanup_data1, inplace=True)
    obj_data1.head(1)
    from sklearn.naive_bayes import MultinomialNB
    multi = MultinomialNB()
    train = []
    classes = []
    for i in obj_data.to_numpy():
        train.append(i[0:len(i)-1])
        classes.append(i[len(i)-1])
    print("obj_df\n",train)
    print("\n class\n", classes)
    multi.fit(train,classes)
    print(multi.predict(obj_data1))
    # print(df.iloc(1))
    # MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    # GaussianNB(priors=None, var_smoothing=1e-09)class_prior_
    dataset = data.to_numpy()
    print(dataset)
    print("The predicted Class is:",dataset[1][len(dataset[0])-1])
    print("The Prior Probability is :",multi.class_prior)
	

	
def main(): 
    trainingSet=[] 
    testSet=[]
    neighbors=[]
    

    handleTrainingDataset("training.txt", trainingSet, testSet)
    handleTestDataset('testing.txt', trainingSet, testSet) 

    predictions=[] 
    k = 3
    print("No of rows in training set:")
    print(len(trainingSet))
    print("\n")
    print("No of rows in test set:")
    print(len(testSet))
    print("\n")
    for x in range(len(testSet)): 
        print("No of training sets to be classified:")
        print(x+1)
        print("\n")
        print("the test set to be classified:")
        print(testSet[x])
        print("\n")
        neighbors = KNearestNeighbors(trainingSet, testSet[x], k) 
    result = getKNN(neighbors)
    print("The result is :")
    print(result)
    predictions.append(result) 
    print('The predicted Class is :' + repr(result))
    mult()
     
main()
