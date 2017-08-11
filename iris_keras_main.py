from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd

numpy.random.seed(5)

dataset = pd.read_csv('iris.csv', header = None)

values = dataset.values[:,0:4].astype(float)
labels_name = dataset.values[:,4:5]
labels = numpy.zeros ( (150,3) )
print labels[:1,].shape

for x in range(0,150):
	if ( labels_name[x]  == 'Iris-setosa') :
		labels[x,0] = 1
	elif ( labels_name[x]  == 'Iris-versicolor') :
		labels[x,1] = 1
	else :
		labels[x,2] = 1


print ( values )
print ( labels )

model = Sequential()
model.add ( Dense ( 12, input_dim = 4, activation ='relu' ) )
model.add ( Dense ( 3, activation = 'softmax' ) )

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit( values, labels, epochs=150, batch_size = 10)

scores = model.evaluate(values, labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print labels.shape

prediction = model.predict(numpy.array([[5.1, 3.5, 1.4, 0.2]]))
print prediction
