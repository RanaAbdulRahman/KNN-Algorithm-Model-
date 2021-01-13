import pandas as pd


  # datasert load
dataset = pd.read_csv("Iris(1).csv")

  #slicing
X_features_input = dataset.iloc[:, :-1].values #features[rows, columms]
#print(X_features_input)
y_label_output = dataset.iloc[:,4].values #labels
       # print(y_label_output)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features_input, y_label_output, test_size=0.20, random_state=4)
        #x_train = 80% of our features data(input)
        #x_test = 20% of our features data(input)
        #y_train = 80% of our lable data(output)
        #y_test = 20 % of pur lable data(output)


        #imported the algorithms from library
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
        # to train the model you have to use the function of "fit()"
        # while traininf we only pass the 80 percent of our data
classifier.fit(X_train, y_train) # X_train = features #y_train= lable
        # now we have to take prediction on testing data
y_pred = classifier.predict(X_test) #here we only pass the features

# for accuracy
from sklearn.metrics import accuracy_score
print('Accuracy Score: ', accuracy_score(y_pred, y_test)) #y_pred is the output
