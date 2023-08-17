import sys
import os
import numpy as np
import pandas as pd
import math
# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)
NUM_FEATS = 90
class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.
		Here, we have provided an example structure for the weights and biases.
		It is a list of weight and bias matrices, in which, the
		dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
		weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
		biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]
		Please note that this is just an example.
		You are free to modify or entirely ignore this initialization as per your need.
		Also you can add more state-tracking variables that might be useful to compute
		the gradients efficiently.
		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''
		self.num_layers = num_layers
		self.num_units = num_units

		self.biases = []
		self.weights = []
		for i in range(num_layers):

			if i==0:
				# Input layer
				self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
			else:
				# Hidden layer
				self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

			self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

		# Output layer
		self.biases.append(np.random.uniform(-1, 1, size=(4, 1)))
		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 4)))
		self.weights=np.array(self.weights)
		self.biases=np.array(self.biases)
	def Relu(self,y1):
		return np.maximum(0,y1)
	def softmax(self,y3):
		y3=np.array(y3)
		y4=[]
		for i in range(len(y3)):
			l=np.exp(y3[i])/np.sum(np.exp(y3[i]))
			l2=[]
			for j in l:
				j=math.ceil(j*10000)
				j=j/10000
				l2.append(j)
			y4.append(l2)
		y4=np.array(y4)
		return y4
	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.
		Note that for a classification task, the output layer should
		be a softmax layer. So perform the computations accordingly
		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''
		X=np.array(X)
		self.y1=np.array(np.dot(X,self.weights[0]))+self.biases[0].T
		#print("y1",self.y1.shape)
		self.y2=self.Relu(self.y1)
		self.y3=np.dot(self.y2,self.weights[1])+self.biases[1].T
		#mean=np.mean(self.y3)
		#std=np.std(self.y3)
		#g4=np.subtract(self.y3,mean)/std
		self.y4=self.softmax(self.y3)
		return self.y4
		#raise NotImplementedError
		
	def derivative(self,pred):
		pred[pred>0]=1
		pred[pred<0]=0
		return pred
	def deriv1(self,y3):
		for i in range(len(y3)):
			s=sum(y3[i])
			l=[]
			for j in range(len(y3[i])):
				l.append(s-y3[i][j])
			y3[i]=l
		return y3
	def output(self,pred):
		ans=[]
		for i in range(len(pred)):
			m=0
			p=0
			for j in range(4):
				if(pred[i][j]>m):
					m=pred[i][j]
					p=j
			ans.append([p])
		ans=np.array(ans)
		return ans
	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)
		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array ,lof shape m x 1
			lamda : Regularization parameter.
		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).
		Hint: You need to do a forward pass before performing backward pass.
		'''
		self.pred=self.__call__(X)
		X=np.array(X)
		y=np.array(y)
		#pre=self.output(self.pred)
		ds=np.subtract(self.pred,y)
		#ds=self.deriv1(self.y3)
		dw2=np.dot(self.y2.T,ds)
		dr=np.dot(ds,self.weights[1].T)
		d1=self.derivative(self.y1)
		d2=np.multiply(dr,d1)
		dw1=np.dot(X.T,d2) 
		dw2=np.array(dw2)
		wt_up=[]
		wt_up.append(dw1)
		wt_up.append(dw2)
		wt_up=np.array(wt_up)
		bs1=np.dot(d2.T,np.ones((X.shape[0],1)))
		bs2=np.dot(ds.T,np.ones((X.shape[0],1)))
		bs_up=[bs1,bs2]
		bs_up=np.array(bs_up)
		return (wt_up,bs_up)
class Optimizer(object):
	'''
	'''
	def __init__(self, learning_rate):
		'''
		Create a Gradient Descent based optimizer with given
		learning rate.
		Other parameters can also be passed to create different types of
		optimizers.
		Hint: You can use the class members to track various states of the
		optimizer.
		'''
		self.learning_rate=learning_rate
		#raise NotImplementedError

	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		'''
		upd_weights=weights-self.learning_rate*delta_weights
		#upd_weights=[[],[]]
		#upd_weights[0]=np.array(np.subtract(weights[0],np.multiply(delta_weights[0],self.learning_rate)))
		#upd_weights[1]=np.array(np.subtract(weights[1],np.multiply(delta_weights[1],self.learning_rate)))
		upd_biases=np.subtract(biases,np.multiply(delta_biases,self.learning_rate))
		#upd_weights=np.array(upd_weights)
		return (upd_weights,upd_biases)
		#raise NotImplementedError

def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.
	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
	Returns
	----------
		MSE loss between y and y_hat.
	'''
	mse=np.square(np.subtract(y,y_hat)).mean()
	return mse
	#raise NotImplementedError

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.
	Parameters
	----------
		weights and biases of the network.
	Returns
	----------
		l2 regularization loss 
	'''
	weights=np.array(weights)
	s=np.square(weights[0]).sum()
	s=s+np.square(weights[1]).sum()
	return s
	#raise NotImplementedError

def loss_fn(y, y_hat, weights, biases, lamda,y2=0):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)
	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter
	Returns
	----------
		l2 regularization loss 
	'''
	loss=loss_mse(y,y_hat)+lamda*loss_regularization(weights,biases)
	return loss
	#raise NotImplementedError

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.
	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
	Returns
	----------
		RMSE between y and y_hat.
	'''
	mse=np.square(np.subtract(y,y_hat)).mean()
	rme=mse**(0.5)
	return rme
	#raise NotImplementedError

def cross_entropy_loss(y, y_hat):
	'''
	Compute cross entropy loss
	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
	Returns
	----------
		cross entropy loss
	'''
	n=len(y)
	y_hat=np.clip(y_hat,1e-16,1)
	s=0
	for i in range(n):
		s=s+(y[i]*math.log(y_hat))
	s=(1/n)*s
	return s
	#raise NotImplementedError
def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.
	Here we have added the code to loop over batches and perform backward pass
	for each batch in the loop.
	For this code also, you are free to heavily modify it.
	'''
	#train_target=np.multiply(train_target,0.001)
	train_input=np.array(train_input)
	train_target=np.array(train_target)
	m = train_input.shape[0]
	# train_input=(train_input-np.min(train_input))/((np.max(train_input)-np.min(train_input)))
	# train_target=(train_target-1922)/(2011-1922)
	#global mean
	#global std
	#global xmean
	#global xstd
	train_input=(train_input-np.mean(train_input))/(np.std(train_input))
	for e in range(max_epochs):
		epoch_loss = 0.
		c=0
		for i in range(0, m, batch_size):
			batch_input = train_input[i:min(i+batch_size,m)]
			batch_target = train_target[i:min(i+batch_size,m)]
			#pred = net(batch_input)
			# Compute gradients of loss w.r.t. weights and biases
			dW, db = net.backward(batch_input, batch_target, lamda) 
			#weights_updated, biases_updated = net.backward(batch_input, batch_target, lamda)       ## this is also edited by 
			# Get updated weights based on current weights and gradients
			weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)
			# Update model's weights and biases
			net.weights = weights_updated
			net.biases = biases_updated
			#print(dW)
			#print(net.weights)
			#c=c+1
			#if(c==30):
				#exit(0)
			# Compute loss for the batch
			pre=net.__call__(batch_input)
			#batch_loss = loss_fn(batch_target, pre, net.weights, net.biases, lamda)
			#print(batch_loss)
			#print(batch_loss)
			#print(net.weights)
			#print(i,batch_loss)
			#epoch_loss += batch_loss
			# print(e, i, rmse(batch_target, pred), batch_loss)

		#print(e, epoch_loss)
		print(e)
		# dev_input=(dev_input-np.min(dev_input))/((np.max(dev_input)-np.min(dev_input)))
		# dev_pred = net.__call__(dev_input)
		# dev_pred=(dev_pred*std)+mean
		# dev_rmse = rmse(dev_target, dev_pred)
		# print("rmse: ",dev_rmse)

		# Write any early stopping conditions required (only for Part 2)
		# Hint: You can also compute dev_rmse here and use it in the early
		# 		stopping condition.
	# After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
	#dev_input=np.array(dev_input)
	#dev_input=(dev_input-np.min(dev_input))/((np.max(dev_input)-np.min(dev_input)))
	#dev_pred = net.__call__(dev_input)
	#dev_pred=(dev_pred*std)+mean
	#dev_pred=np.multiply(dev_pred,1000)
	#dev_rmse = rmse(dev_target, dev_pred)
	#dev_target=np.array(dev_target)
	#dev_pred=np.array(dev_pred)
	#print("rmse: ",dev_rmse)
	#rse=dev_rmse
	#print('RMSE on dev data: {:.5f}'.format(dev_rmse))


def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.
	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d
	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	# inputs=(inputs-np.min(inputs))/((np.max(inputs)-np.min(inputs)))
	inputs=(inputs-np.mean(inputs))/(np.std(inputs))
	out_pred=net.__call__(inputs)
	#mean=np.mean(train_target)
	#std=np.std(train_target)
	#out_pred=(out_pred*std)+mean
	#print("std is "+str(std))
	#print("mean is "+str(mean))
	#print(out_pred)
	l1=[]
	mi=2100
	ma=0
	print("Id,Predictions")
	c=1
	for i in out_pred:
		po=0
		m=0
		for j in range(4):
			if(i[j]>m):
				m=i[j]
				po=j
		if(po==0):
			ans="Very Old"
		elif(po==1):
			ans="Old"
		elif(po==2):
			ans="Recent"
		else:
			ans="New"	
		print(str(c)+","+ans)
		c=c+1
	#raise NotImplementedError
def read_data():
	'''
	Read the train, dev, and test datasets
	'''
	df = pd.read_csv("train.csv")
	train_input = df.iloc[0:,1:]
	train_target=df.iloc[0:,0:1]
	df=pd.read_csv("dev.csv")
	dev_input = df.iloc[0:,1:]
	dev_target=df.iloc[0:,0:1]
	df=pd.read_csv("test.csv")
	test_input = df.to_numpy()
	return (train_input, train_target, dev_input, dev_target, test_input)

def one_hot(vector):
		one_hot_vector=[]
		for i in range(len(vector)):
			l=[0]*4
			l[vector[i][0]]=1
			one_hot_vector.append(l)
		one_hot_vector=np.array(one_hot_vector)
		return one_hot_vector
def main():

	# Hyper-parameters 
	max_epochs = 400 # 400 best 
	batch_size = 256  # 256
	learning_rate = 0.0001 # 0.0001
	num_layers = 1 # 1
	num_units = 64 #64
	lamda = 0.8 # Regularization Parameter 0.8

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	train_target=np.array(train_target)
	for i in range(len(train_target)):
		if(train_target[i][0]=="Very Old"):
			train_target[i][0]=0
		elif(train_target[i][0]=="Old"):
			train_target[i][0]=1
		elif(train_target[i][0]=="Recent"):
			train_target[i][0]=2
		else:
			train_target[i][0]=3
	train_target=one_hot(train_target)
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	get_test_data_predictions(net, test_input)


if __name__ == '__main__':
	main()
