// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// RBM specification

package rbm

import (
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

type trainParameters struct {
	learning_rate        WeightT //equivalent to the \theta in the Delta Rule
	regularization_rate  WeightT //equivalent to the \lambda in the Delta Rule
	momentum_rate        WeightT //equivalent to the \mu in the Delta Rule
	gen_learn_importance WeightT //equivalent to the \alpha in the hybrid learning equation
	gibbs_chain_length   int     //the k value of CD-k
}

// RBMTrainer specifiers how an RBM should be trained.
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
