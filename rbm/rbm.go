// Copyright Weidoliang (2013)

// Package rbm provides and implementation of the SparseClassRBM for 
//	classification/regression.
// SparseClassRBM:
// E(y, X, h) = -SumOver_0<=i<C_{h^T . W(i) . e_x_i + b(i) . e_x_i} - h^T . c - d . y - h^T . U . y
// p(y, X, h) = exp(-E(y,X,h))/Z
//
// Reference:
// 	[1]. Hinton, 2010, A Practical Guide to Training Restricted Boltzmann Machines (Ver. 1) 
//
// TODO:
//	initialization of random seed.
//  one visible bias per feature class VS one visible bias per visible units
//
package rbm

// RBM Object for storing the parameters of a gven SparseClassRBM
type SparseClassRBM struct {
	w           [][][]float32 //interactions between X and h [feature_class, hidden, visible]
	b           [][]float32   //bias of X
	c           []float32   //bias of h
	u           []float32   //interactions between y and h
	d           float32     //bias of y
	x_class_num uint32      //number of Classes in X
	h_num       uint32      //number of hidden units
}

// DataInstance is used for storing data sample for training and prediction.
// In the case of prediction, the value of y is ignored.
type DataInstance struct {
	x []uint32 //values of each classes, in the order of Class0, Class1, ...
	y float32  //values of Y
}


type TrainParameters struct {
	learning_rate        float32 //equivalent to the \theta in the Delta Rule
	regularization_rate  float32 //equivalent to the \lambda in the Delta Rule
	momentum_rate        float32 //equivalent to the \mu in the Delta Rule
	gen_learn_importance float32 //equivalent to the \alpha in the hybrid learning equation
}

type RBMTrainer struct {
	rbm             *SparseClassRBM  //RBM model
	parameters      *TrainParameters //Training parameters
	training_data   []DataInstance   //Training data
	validation_data []DataInstance   //Test data
}


func init() {
	//TODO(weidoliang): add random seed initialization
}