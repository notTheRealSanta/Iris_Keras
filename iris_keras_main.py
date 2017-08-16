from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd

# init a random seed
numpy.random.seed(5)

#reading iris dataset with pandas
dataset = pd.read_csv('iris.csv', header = None)

#Categorising the datesets to values and labels for training

values = dataset.values[:,0:4].astype(float)
labels_name = dataset.values[:,4:5]
labels = numpy.zeros ( (150,3) )

#one-hot encoding
for x in range(0,150):
	if ( labels_name[x]  == 'Iris-setosa') :
		labels[x,0] = 1
	elif ( labels_name[x]  == 'Iris-versicolor') :
		labels[x,1] = 1
	else :
		labels[x,2] = 1

#taking the first row as test_data
test_data = values[0]

values = numpy.delete ( values, (0), axis = 0)
labels = numpy.delete ( labels , (0), axis = 0)

#print ( values )
#print ( labels )

#building the model
model = Sequential()
model.add ( Dense ( 12, input_dim = 4, activation ='relu' ) )
model.add ( Dense ( 3, activation = 'softmax' ) )

#compling ( setup of the model )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit trains the model
model.fit( values, labels, epochs=150, batch_size = 10)

#evaluate returns the loss and mertric values ( i.e here the accuracy)
scores = model.evaluate(values, labels)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#prediction
print ("Prediction for : ")
print (test_data)
prediction = model.predict(numpy.array([test_data]))
print (prediction)
