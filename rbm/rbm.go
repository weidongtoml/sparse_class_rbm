// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rbm provides and implementation of the SparseClassRBM for
// classification/regression.
//
// SparseClassRBM:
//  E(y, X, h) = -SumOver_0<=i<C_{h^T . W(i) . e_x_i + b(i) . e_x_i} - h^T . c - d . y - h^T . U . y
//  p(y, X, h) = exp(-E(y,X,h))/Z
//
// Reference:
//  [1]. Hinton, 2010, A Practical Guide to Training Restricted Boltzmann Machines (Ver. 1)
//
// TODO:
//  Initialization of random seed.
//  One visible bias per feature class VS one visible bias per visible units
//
package rbm

import (
	_ "fmt"
	_ "math"
	"math/rand"
	"time"
)

type WeightT float64

// RBM Object for storing the parameters of a gven SparseClassRBM
type SparseClassRBM struct {
	w             [][][]WeightT //interactions between X and h [feature_class, hidden, visible]
	b             [][]WeightT   //bias of X
	c             []WeightT     //bias of h
	u             []WeightT     //interactions between y and h
	d             WeightT       //bias of y
	x_class_num   int           //number of Classes in X
	x_class_sizes []int         //Size of each classes
	h_num         int           //number of hidden units
}

// DataInstance is used for storing data sample for training and prediction.
// In the case of prediction, the value of y is ignored.
type DataInstance struct {
	x []int //values of each classes, in the order of Class0, Class1, ...
	y int   //values of Y
}

type DataInstanceAccessor interface {
	Reset()
	NextInstance() DataInstance
}

type trainParameters struct {
	learning_rate        WeightT //equivalent to the \theta in the Delta Rule
	regularization_rate  WeightT //equivalent to the \lambda in the Delta Rule
	momentum_rate        WeightT //equivalent to the \mu in the Delta Rule
	gen_learn_importance WeightT //equivalent to the \alpha in the hybrid learning equation
	gibbs_chain_length   int     //the k value of CD-k
}

type RBMTrainer struct {
	rbm                      *SparseClassRBM      //RBM model
	prev_delta               *SparseClassRBM      //For storing previous Delta
	parameters               trainParameters      //Training parameters
	training_data_accessor   DataInstanceAccessor //Training data
	validation_data_accessor DataInstanceAccessor //Test data
}

func init() {
	rand.Seed(time.Now().Unix())
}
