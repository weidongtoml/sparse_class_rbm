// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A collection of functions and methods for evaluating the given model.

package rbm

type BinaryClassifier interface {
	GetPrediction(instance *DataInstance) WeightT
}

type Coordinate struct {
	x float64
	y float64
}

// ROCAuc calcuates the area under the receiver operating curve.
func ROCAuc(classifier *BinaryClassifier, data_accessor *DataInstanceAccessor) float64 {
	return 0
}

// ROC returns the coordinate of the Receiver Operating Curve based based on the
// given
func ROC(classifier *BinaryClassifier, data_accessor *DataInstanceAccessor) []Coordinate {
	return nil
}

// RMSE returns the root mean squared error.
func RMSE(classifier *BinaryClassifier, data_accessor *DataInstanceAccessor) float64 {
	return 0
}

// LogLikelihood returns the log likelihood of the classifier fitting the given data.
func LogLikelihood(classifier *BinaryClassifier, data_accessor *DataInstanceAccessor) float64 {
}

// Sparsity returns the total sparsity of the parameters of the given RBM.
func (rbm *SparseClassRBM) Sparsity() float64 {
	return 0
}

// L2NormOfParamaeters returns the L2 norm of all the parameters of the given RBM.
func (rbm *SparseClassRBm) L2NormOfParamaeters() float64 {
	return 0
}

// TODO(weidoliang): add W, U, specific sparsity and L2 norm measures.
