"""
Developer : Naveen
This file contains the code to create a random forests model. Model is tuned manually with out any libraries.
"""

from sklearn.ensemble import RandomForestRegressor
import pandas
import numpy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from math import sqrt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler




def RunModel(trait,estimator,samplesplit,max_depth,max_feature):
    
	#reading the data
	df = pandas.read_csv(r'DataSet.csv')
    df= df.fillna(value=0)
    
	#getting the input and output variables
    X = df.loc[:,'mediaUsage':'Scheduling_OfficeTools_Weather']
    Y = df.loc[:,trait]
	
	#creating the RandomForest model. Scalar and models are added to the pipeline
    rf = RandomForestRegressor(n_estimators=estimator,bootstrap=True,
                        min_samples_split=samplesplit,max_features=max_feature,criterion='mse',max_depth=max_depth)
    seed=7
    numpy.random.seed(seed)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('RandomForest', rf))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=4, random_state=seed)


    results = cross_val_predict(pipeline,X,Y,cv=kfold)
    
	
	#getting the values for all days and aggregating them for each participant
    df['predicted'] = results
    grouped = df.groupby(['ID'])
    outputlist=[(key,numpy.average(value[trait]),numpy.average(value['predicted'])) for (key,value) in grouped.__iter__()]
    outputdf= pandas.DataFrame(outputlist,columns=['ID','Actual','Predicted'])

    #plotting the fit line for results and actual values
    fig, ax = plt.subplots()
    ax.scatter(outputdf['Actual'],outputdf['Predicted'], edgecolors=(0, 0, 0))
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlabel('Actual Values (Big Five PreSurvey)')
    ax.set_ylabel('Predicted Values')
    
    plt.show()
    #plt.savefig('/home/naveen/Desktop/RandomForests_Plots/'+trait+': estimator'+str(estimator)+':'+': Sampleleaf'+str(sampleleaf)+':maxdepth'+str(maxDepth)+'MaxFeatures'+str(max_feature)+'.png')
    plt.close()

    return (sqrt(mean_squared_error(outputdf['Actual'],outputdf['Predicted']))*100)





traits=['Openness','Conscientiousness','Extraversion','Agreeableness','Neuroticism']

estimators=[3,5,10,20,40,60,80,100,200]
SamplesLeaf=[20,50,100]
maxDepths=[5,10,20,30,40,50,60,70,100]
maxFeatures=[5,6,8,10]

output={}
for trait in traits:
    for estimator in estimators:

        for maxDepth in maxDepths:
            for max_feature in maxFeatures:
               

                for sampleleaf in SamplesLeaf:
                    results=[]
                    for i in range(500):
                        results.append(RunModel(trait,estimator,sampleleaf,maxDepth,max_feature))

                    Modelparameters=trait+":  estimators-"+str(estimator)+" "+'Max Depth-'+str(maxDepth)+" "+"Sample Leaf, Minsamples to split"+str(sampleleaf)+'features'+str(max_feature)
                    output[Modelparameters]  = numpy.average(results)

for key,val in output.items():
    print(key,"-------",val)
