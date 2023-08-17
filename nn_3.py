import sys
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 71
# mean=1
# std=1
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
		self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))
		self.weights=np.array(self.weights)
		self.biases=np.array(self.biases)
	def Relu(self,y1):
		return np.maximum(0,y1)
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
		return self.y3
		#raise NotImplementedError
		
	def derivative(self,pred):
		pred[pred>0]=1
		pred[pred<0]=0
		return pred
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
		dw2=np.dot(self.y2.T,2*np.subtract(self.pred,y))+2*lamda*self.weights[1]
		a=2*np.subtract(self.pred,y)
		dr=np.dot(a,self.weights[1].T)
		d1=self.derivative(self.y1)
		d2=np.multiply(dr,d1)
		dw1=np.dot(X.T,d2) + 2*lamda*self.weights[0]
		dw2=np.array(dw2)
		wt_up=[]
		wt_up.append(dw1)
		wt_up.append(dw2)
		wt_up=np.array(wt_up)
		bs1=np.dot(d2.T,np.ones((X.shape[0],1)))
		bs2=np.dot(a.T,np.ones((X.shape[0],1)))
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
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

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
	x=pd.DataFrame(train_input)
	x=x.corr().abs()
	utri=x.where(np.triu(np.ones(x.shape),k=1).astype(np.bool))
	# print(utri)
	drops=[c for c in utri.columns if any(utri[c]>0.5)]
	# print(len(drops))
	# print(drops)
	global i_list
	i_list=[i for i in range(0,89)]
	h=[]
	for k in i_list:
		if(k not in drops):
			h.append(k)
	i_list=h
	# print(i_list)
	# print(len(i_list))
	# exit(0)
	l=[]
	for i in i_list:
			l.append(train_input[0:,i])
	l=np.array(l)
	train_input=l.T
	# print(train_input.shape)
	# corr_mat=np.corrcoef(train_input)
	

	# print("corr dim"+str(x.shape))
	 # print(x)
	# cor_pairs=x.unstack()
	# s_pairs=cor_pairs.sort_values(kind="quicksort")
	# high_corr=s_pairs[abs(s_pairs)>0.57]
	# high_corr=high_corr[abs(high_corr)<1.0]
	# print(high_corr)
	# # print(high_corr.shape)
	# s=np.dot(train_input.T,train_input)
	# print(s.shape)
	# print("condition number is "+str(np.linalg.cond(x)))
	# exit(0)
	train_target=np.array(train_target)
	m = train_input.shape[0]
	# train_input=(train_input-np.min(train_input))/((np.max(train_input)-np.min(train_input)))
	# train_target=(train_target-1922)/(2011-1922)
	global mean
	global std
	global xmean
	global xstd
	train_input=(train_input-np.mean(train_input))/(np.std(train_input))
	mean=np.mean(train_target)
	std=np.std(train_target)
	train_target=np.subtract(train_target,mean)/std
	p=np.random.permutation(m)
	tr_in=[]
	tr_tr=[]
	for i in p:
		tr_in.append(train_input[i])
		tr_tr.append(train_target[i])
	train_input=np.array(tr_in)
	train_target=np.array(tr_tr)
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
			batch_loss = loss_fn(batch_target, pre, net.weights, net.biases, lamda)
			#print(batch_loss)
			#print(batch_loss)
			#print(net.weights)
			#print(i,batch_loss)
			epoch_loss += batch_loss
			# print(e, i, rmse(batch_target, pred), batch_loss)

		#print(e, epoch_loss)
		# dev_input=(dev_input-np.min(dev_input))/((np.max(dev_input)-np.min(dev_input)))
		# dev_pred = net.__call__(dev_input)
		# dev_pred=(dev_pred*std)+mean
		# dev_rmse = rmse(dev_target, dev_pred)
		# print("rmse: ",dev_rmse)

		# Write any early stopping conditions required (only for Part 2)
		# Hint: You can also compute dev_rmse here and use it in the early
		# 		stopping condition.

	# After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
	dev_input=np.array(dev_input)
	l=[]
	for i in i_list:
			l.append(dev_input[0:,i])
	l=np.array(l)
	dev_input=l.T
	dev_input=(dev_input-np.min(dev_input))/((np.max(dev_input)-np.min(dev_input)))
	dev_pred = net.__call__(dev_input)
	dev_pred=(dev_pred*std)+mean
	#dev_pred=np.multiply(dev_pred,1000)
	dev_rmse = rmse(dev_target, dev_pred)
	print("rmse: ",dev_rmse)
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
	inputs=np.array(inputs)
	l=[]
	for i in i_list:
			l.append(inputs[0:,i])
	l=np.array(l)
	inputs=l.T
	inputs=(inputs-np.mean(inputs))/(np.std(inputs))
	out_pred=net.__call__(inputs)
	# mean=np.mean(train_target)
	# std=np.std(train_target)
	out_pred=(out_pred*std)+mean
	#print("std is "+str(std))
	#print("mean is "+str(mean))
	# print(out_pred)
	l1=[]
	mi=2100
	ma=0
	print("Id,Predictions")
	c=1
	for i in out_pred:
		print(str(c)+","+str(min(round(i[0]),2011)))
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


def main():

	# Hyper-parameters 
	max_epochs = 1200 # 1200 best 
	batch_size = 256  # 256
	learning_rate = 0.00001 # 0.00001
	num_layers = 1 # 1
	num_units = 85 #85
	lamda = 0.8 # Regularization Parameter 0.8

	train_input, train_target, dev_input, dev_target, test_input = read_data()
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
