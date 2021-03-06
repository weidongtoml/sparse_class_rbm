// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	_ "fmt"
	"testing"
)

func Test_Sigmoid(t *testing.T) {
	test_cases := []struct {
		x WeightT
		y WeightT
	}{
		{0.0, 0.5},
		{0.5, 0.62245931},
		{1.0, 0.73105857},
		{10.0, 0.99995460},
	}
	for i, t_case := range test_cases {
		y := Sigmoid(t_case.x)
		if !EqualWithinPrecision(t_case.y, y, kPrecision) {
			t.Errorf("TestCase #%d: Expected Sigmoid(%v) = %v, but got %v.", i, t_case.x, t_case.y, y)
		}
	}
}

func Test_SoftPlus(t *testing.T) {
	test_cases := []struct {
		x WeightT
		y WeightT
	}{
		{0.0, 0.69314718},
		{0.5, 0.97407698},
		{1.0, 1.31326168},
	}
	for i, t_case := range test_cases {
		y := SoftPlus(t_case.x)
		if !EqualWithinPrecision(t_case.y, y, kPrecision) {
			t.Errorf("TestCase #%d: Expected Softplus(%v) = %v, but got %v.", i, t_case.x, t_case.y, y)
		}
	}
}

func Test_DotProduct(t *testing.T) {
	test_cases := []struct {
		a []WeightT
		b []WeightT
		p WeightT
	}{
		{
			[]WeightT{1.0, 2.0, 3.0, 4.0},
			[]WeightT{1.0, 2.0, 3.0, 4.0},
			1.0*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0,
		}, {
			[]WeightT{1.0, 2.0, 3.0, 4.0},
			[]WeightT{4.0, 3.0, 2.0, 1.0},
			1.0*4.0 + 2.0*3.0 + 3.0*2.0 + 4.0*1.0,
		}, {
			[]WeightT{1.0, 2.0, 3.0, 4.0},
			[]WeightT{0.0, 0.0, 0.0, 0.0},
			0.0,
		},
	}
	for i, t_case := range test_cases {
		p := DotProduct(t_case.a, t_case.b)
		if !EqualWithinPrecision(t_case.p, p, kPrecision) {
			t.Errorf("TestCase #%d: Expected Dot(%v, %v) = %v, but got %v.", i, t_case.a, t_case.b, t_case.p, p)
		}
	}
}

func Test_ScalarProduct(t *testing.T) {
	test_cases := []struct {
		v     []WeightT
		s     WeightT
		exp_v []WeightT
	}{
		{
			[]WeightT{0, 1, 2, 3, 4, 5},
			0,
			[]WeightT{0, 0, 0, 0, 0, 0},
		}, {
			[]WeightT{0, 1, 2, 3, 4, 5},
			1,
			[]WeightT{0, 1, 2, 3, 4, 5},
		}, {
			[]WeightT{0, 1, 2, 3, 4, 5},
			2,
			[]WeightT{0, 2, 4, 6, 8, 10},
		}, {
			[]WeightT{0, 1, 2, 3, 4, 5},
			-2,
			[]WeightT{0, -2, -4, -6, -8, -10},
		},
	}
	for i, t_case := range test_cases {
		ScalarProduct(t_case.v, t_case.s)
		if !ArraysEqualWithinPrecision(t_case.v, t_case.exp_v, kPrecision) {
			t.Errorf("TestCase: %d: error.", i)
		}
	}
}
