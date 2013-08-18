// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"testing"
)

func getSampleRBMForProbabilityTest() *SparseClassRBM {
	number_hidden_units := 4
	visible_class_sizes := []int{1, 2, 3}
	// Biases of X
	visible_class_biases := [][]WeightT{
		{0.1},
		{0.2, 0.3},
		{0.4, 0.05, -0.2}}
	// Bias of Y
	positive_y_bias := WeightT(0.03)
	// Biases of H
	h_biases := []WeightT{
		0.10, 0.11, 0.12, 0.13,
	}
	// Interaction of X and H
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
		{0, 2, 2, 0.06},
		//Hidden Unit 1
		{1, 0, 0, 0.11}, //X Class 0
		{1, 1, 0, 0.12}, //X Class 1
		{1, 1, 1, 0.13},
		{1, 2, 0, 0.14}, //X Class 2
		{1, 2, 1, 0.15},
		{1, 2, 2, 0.16},
		//Hidden Unit 2
		{2, 0, 0, 0.21}, //X Class 0
		{2, 1, 0, 0.22}, //X Class 1
		{2, 1, 1, 0.23},
		{2, 2, 0, 0.24}, //X Class 2
		{2, 2, 1, 0.25},
		{2, 2, 2, 0.26},
		//Hidden Unit 3, All are zero
		{3, 0, 0, 0.00}, //X Class 0
		{3, 1, 0, 0.00}, //X Class 1
		{3, 1, 1, 0.00},
		{3, 2, 0, 0.00}, //X Class 2
		{3, 2, 1, 0.00},
		{3, 2, 2, 0.00},
	}
	// Interactions of Y and H
	u_weights := []WeightT{
		0.01, 0.02, 0.03, 0.04,
	}

	rbm := new(SparseClassRBM)
	rbm.Initialize(visible_class_sizes, visible_class_biases, number_hidden_units, positive_y_bias)
	for _, w := range w_weights {
		rbm.SetW(w.h_index, w.c_index, w.c_value, w.v)
	}
	for j, u := range u_weights {
		rbm.SetU(j, u)
	}
	for j, h := range h_biases {
		rbm.SetC(j, h)
	}
	return rbm
}

// Test the calculation of P(h | X, Y).
func Test_probDistOfHGivenXY(t *testing.T) {
	test_cases := []struct {
		x []int
		y int
		h []WeightT
	}{
		{
			[]int{0, 0, 0},
			0,
			[]WeightT{Sigmoid(0.17), Sigmoid(0.48), Sigmoid(0.79), Sigmoid(0.13)},
		}, {
			[]int{0, 1, 2},
			1,
			[]WeightT{Sigmoid(0.21), Sigmoid(0.53), Sigmoid(0.85), Sigmoid(0.17)},
		},
	}

	rbm := getSampleRBMForProbabilityTest()
	h := make([]WeightT, rbm.SizeOfHiddenLayer())
	for i, t_case := range test_cases {
		rbm.probDistOfHGivenXY(h, t_case.x, t_case.y)
		if !ArraysEqualWithinPrecision(t_case.h, h, kPrecision) {
			t.Errorf("TestCase: #%d, expected \n%v but got\n %v.", i, t_case.h, h)
		}
	}
}

// Test the calculation of P(X_c | h).
func Test_probOfXInClassCGivenH(t *testing.T) {
	test_cases := []struct {
		h   []WeightT
		p_x []WeightT
		c   int
	}{
		{
			[]WeightT{0.1, 0.2, 0.3, 0.4},
			[]WeightT{1},
			0,
		}, {
			[]WeightT{1, 0, 0, 1},
			[]WeightT{Exp(0.02) / (Exp(0.02) + Exp(0.03)), Exp(0.03) / (Exp(0.02) + Exp(0.03))},
			1,
		}, {
			[]WeightT{0.4, 0.3, 0.2, 0.1},
			[]WeightT{
				Exp(0.106) / (Exp(0.106) + Exp(0.115) + Exp(0.124)),
				Exp(0.115) / (Exp(0.106) + Exp(0.115) + Exp(0.124)),
				Exp(0.124) / (Exp(0.106) + Exp(0.115) + Exp(0.124)),
			},
			2,
		},
	}
	rbm := getSampleRBMForProbabilityTest()
	for i, t_case := range test_cases {
		p_x := rbm.probOfXInClassCGivenH(t_case.c, t_case.h)
		if !ArraysEqualWithinPrecision(t_case.p_x, p_x, kPrecision) {
			t.Errorf("TestCase: #%d, expected \n%v but got\n %v.", i, t_case.p_x, p_x)
		}
	}
}

// Test the calculation of P(Y=1 | X).
func Test_probOfYGivenX(t *testing.T) {
	test_cases := []struct {
		x []int
		y WeightT
	}{
		{
			[]int{0, 0, 0},
			(Exp(0.03+SoftPlus(0.18)+SoftPlus(0.50)+SoftPlus(0.82)+SoftPlus(0.17)) /
				(Exp(0.03+SoftPlus(0.18)+SoftPlus(0.50)+SoftPlus(0.82)+SoftPlus(0.17)) +
					Exp(SoftPlus(0.17)+SoftPlus(0.48)+SoftPlus(0.79)+SoftPlus(0.13)))),
		}, {
			[]int{0, 1, 2},
			(Exp(0.03+SoftPlus(0.21)+SoftPlus(0.53)+SoftPlus(0.85)+SoftPlus(0.17)) /
				(Exp(0.03+SoftPlus(0.21)+SoftPlus(0.53)+SoftPlus(0.85)+SoftPlus(0.17)) +
					Exp(SoftPlus(0.20)+SoftPlus(0.51)+SoftPlus(0.82)+SoftPlus(0.13)))),
		},
	}

	rbm := getSampleRBMForProbabilityTest()
	for i, t_case := range test_cases {
		p_y := rbm.probOfYGivenX(t_case.x)
		if !EqualWithinPrecision(t_case.y, p_y, kPrecision) {
			t.Errorf("TestCase: #%d, expected %v but got %v.", i, t_case.y, p_y)
		}
	}
}

func Test_wHDotX(t *testing.T) {
	test_cases := []struct {
		h_index int
		x       []int
		p       WeightT
	}{
		{
			0,
			[]int{0, 0, 0},
			0.07,
		}, {
			0,
			[]int{0, 1, 2},
			0.10,
		}, {
			3,
			[]int{0, 1, 0},
			0.0,
		}, {
			3,
			[]int{0, 1, 1},
			0.0,
		},
	}

	rbm := getSampleRBMForProbabilityTest()
	for i, t_case := range test_cases {
		p := rbm.wHDotX(t_case.h_index, t_case.x)
		if !EqualWithinPrecision(t_case.p, p, kPrecision) {
			t.Errorf("TestCase #%d: expected %v but got %v.", i, t_case.p, p)
		}
	}
}
