// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"testing"
)

func getSampleRBMForProbabilityTest() *SparseClassRBM {
	var rbm SparseClassRBM
	num_visible_classes := 3
	feature_classes := []int{1, 2, 3}
	visible_class_biases := [][]WeightT{
		{0.1},
		{0.2, 0.3},
		{0.4, 0.05, -0.2}}
	number_hidden_units := 4
	positive_y_bias := WeightT(0.03)
	w_weights := []struct {
		h_index int
		c_index int
		c_value int
		v       WeightT
	}{
		//Hidden Unit 0
		{0, 0, 0, 0.01}, //X Class 0
		{0, 1, 0, 0.02}, //X Class 1
		{0, 1, 1, 0.03},
		{0, 2, 0, 0.04}, //X Class 2
		{0, 2, 1, 0.05},
		{0, 3, 2, 0.06},
		//Hidden Unit 1
		{1, 0, 0, 0.11}, //X Class 0
		{1, 1, 0, 0.12}, //X Class 1
		{1, 1, 1, 0.13},
		{1, 2, 0, 0.14}, //X Class 2
		{1, 2, 1, 0.15},
		{1, 3, 2, 0.16},
		//Hidden Unit 2
		{2, 0, 0, 0.21}, //X Class 0
		{2, 1, 0, 0.22}, //X Class 1
		{2, 1, 1, 0.23},
		{2, 2, 0, 0.24}, //X Class 2
		{2, 2, 1, 0.25},
		{2, 3, 2, 0.26},
		//Hidden Unit 3, All are zero
	}

	rbm := new(SparseClassRBM)
	rbm.Initialize(num_visible_classes, visible_class_biases, number_hidden_units, positive_y_bias)
	for _, w := range w_weights {

	}
	return &rbm
}

func Test_probDistOfHGivenXY(t *testing.T) {

}
