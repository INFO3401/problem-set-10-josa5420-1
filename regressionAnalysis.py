""""from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import r2_score
import pandas as pd
class AnalysisData:
    def __init__(self, dataset):
        self.dataset = dataset   #Which holds the parsed dataset
        self.variables = [i for i in self.dataset.columns if i != 'competitorname']#.remove('competitorname') #which will hold a list containing the indexes for all of the variables in your data
        self.targetY = ""
    
    def setTargetY(self, a):
        self.targetY = a
    
class LinearAnalysis: #will contain your functions for doing linear regression
    def __init__(self, targetY):
        self.bestX = 0#bestX #holds the best X predictor for data
        self.targetY = targetY#targetY #holds the index to the target dependent variable
        self.fit = 0 #fit  #will hold how well bestX predicts target variable
    def runSimpleAnalysis(self,t):
        acc = 0
        var = ""
        test = [i for i in t.variables if i != self.targetY]
        for i in test:
            regr = LinearRegression()
            regr.fit(t.dataset[i].reshape(-1,1),t.dataset[self.targetY].reshape(-1,1))
            predict = regr.predict(t.dataset[i].reshape(-1,1))
            a = r2_score(t.dataset[self.targetY].reshape(-1,1), predict)
            if a > acc:
                acc = a
                var = i
        print(str(var)+" fit: "+str(acc))
        print("coef: "+str(regr.coef_))
        print("intercept: "+str(regr.intercept_))

        #print(regr.intercept_[0])


        
    
class LogisticAnalysis: #will contain your functions for doing logistic regression
    def __init__(self, targetY):
        self.bestX = 0#bestX #holds the best X predictor for data
        self.targetY = targetY#targetY #holds the index to the target dependent variable
        self.fit = 0#fit  #will hold how well bestX predicts target variable
    def runSimpleAnalysis(self,t):
        acc = 0
        var = ""
        test = [i for i in t.variables if i != self.targetY]
        for i in test:
            regr = LogisticRegression()
            regr.fit(t.dataset[i].astype(int).reshape(-1,1),t.dataset[self.targetY].astype(int).reshape(-1,1))
            predict = regr.predict(t.dataset[i].reshape(-1,1))
            a = r2_score(t.dataset[self.targetY].reshape(-1,1), predict)
            if a > acc:
                acc = a
                var = i
        print(str(var)+" fit: "+str(acc))
        print("coef: " + str(regr.coef_))
        print("intercept: "+str(regr.intercept_))

        
    def runMultipleRegression(self,t):
        acc = 0
        var = ""
        test = [i for i in t.variables if i != self.targetY]        
        regr = LogisticRegression()
        regr.fit(t.dataset[test],t.dataset[self.targetY].reshape(-1,1))
        predict = regr.predict(t.dataset[test])
        print("fit: "+str(r2_score(t.dataset[self.targetY], predict)))
        print("coefs: "+str(regr.coef_))
        print("intercept: "+str(regr.intercept_))

df = pd.read_csv('candy-data.csv')
AD1 = AnalysisData(df)

AD1.setTargetY('chocolate')
lin = LinearAnalysis(AD1.targetY)
lin.runSimpleAnalysis(AD1)
AD1.setTargetY('chocolate')
log = LogisticAnalysis(AD1.targetY)
log.runSimpleAnalysis(AD1)

#1. Add a function to the LogisticAnalysis object called runSimpleAnalysis. 
#This function should take in an AnalysisData object as a parameter and should use 
#this object to compute which variable best predicts whether or not a candy is chocolate 
#using logistic regression. Print the variable name and the resulting fit. Do the two 
#functions find the same optimal variable? Which method best fits this data? Make sure
# your best predictor is NOT the same as the targetY variable. 

#runSimpleAnalysis for both linear and logistic regression find the same variable, 'fruity', optimal for
#predicting whether a candy is chocolate or not. Linear regression has a better fit, at ~.5502

#2. Add a function to the LogisticAnalysis object called runMultipleRegression. This function 
#should take in an AnalysisData object as a parameter and should use this object to compute a 
#multiple logistic regression using all of the possible independent variables in your dataset 
#to predict whether or not a candy is chocolate (note, you should not use your dependent variable 
#as an independent variable). Print the variable name and resulting fit. In your testing code, 
#create a new LogisticAnalysis object and use it to run this function on your candy data. Compare 
#the outcomes of this and the simple logistic analysis. Which model best fits the data? Why? 

log.runMultipleRegression(AD1)

#runMultipleRegression found a fit of ~.7607, which is the highest of the three regressions. This is
#because multiple regression allows for multiple variables to be considered in the analysis, which happens
#to be beneficial in this case.


#3. Write the equations for your linear, logistic, and multiple logistic regressions. Hint: Use the equations
#from the slides from Monday's lecture to work out what a logistic regression equation might look like. 
#The coef_ and intercept_ attributes of your regression object will help a lot here!

#From the information printed in runSimpleAnalysis and runMultipleRegression:
#linear regression:      y = -.6503 + .0216 x
#logistic regression:    y = -3.0656 + 0.0591 x
#multiple regression:    y = -1.6826 - 2.52858047 x1 - 0.19697876 x2 + 0.03940308 x3 - 0.16539952 x4
#                        + 0.49783674 x5 - 0.47591613 x6 + 0.81511886 x7 - 0.59971553 x8 - 0.2581028 x9  
#                        + 0.3224988 x10 + 0.05387906 x11






#__________________________FRIDAY__________________________

#4. Identify the independent variable(s) and its type (e.g., categorical, continuous, or discrete), 
#the dependent variable and its type, and the null hypothesis for each of the following scenarios: 

#(a) What candies contain more sugar, those with caramel or those with chocolate?
#      Independent variable: Type of Candy
#      Dependent variable:   Amount of Sugar
#      Null Hypothesis:      Both Caramel and Chocolate candies contain the same amount of sugar.

#(b) Are there more split ticket voters in blue states or red states? 
#      Independent variable: Color of state
#      Dependent variable:   Amount of split ticket voters
#      Null Hypothesis:      Red and Blue states have the same amount of split ticket voters.


#(c) Do phones with longer battery life sell at a higher or lower rate than other phones?
#      Independent variable: Type of phone
#      Dependent variable:   Selling rate
#      Null Hypothesis:      Phones with longer battery life sell at the same rate as other phones.


""""