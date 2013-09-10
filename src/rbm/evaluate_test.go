// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	_ "reflect"
	"testing"
)

func Test_AUC(t *testing.T) {
	test_case := []*Coordinate{
		{
			1, 5, 0.1,
		}, {
			2, 6, 0.3,
		}, {
			3, 2, 0.33,
		}, {
			5, 1, 0.50,
		}, {
			7, 0, 0.90,
		},
	}
	expected_area := float64(0.873016)

	//coordinates := Coordinates{test_case}
	actual_area := AUC(test_case)
	if !EqualWithinPrecesionF64(actual_area, expected_area, kPrecision) {
		t.Errorf("Expected area to be %f but got %f.", expected_area, actual_area)
	}
}

func Test_Sparsity(t *testing.T) {
	test_cases := []struct {
		w           interface{}
		z           interface{}
		z_count     int
		total_count int
	}{
		{
			0,
			0,
			1,
			1,
		}, {
			[]WeightT{1, 2, 3, 4, 0, 0, 0},
			WeightT(0),
			3,
			7,
		}, {
			[]int{1, 2, 3, 4, 0, 0, 0},
			1,
			1,
			7,
		}, {
			[][]int{
				[]int{1, 2, 3, 0},
				[]int{0, 1, 2},
				[]int{},
			},
			0,
			2,
			7,
		}, {
			[][][]int{
				[][]int{
					[]int{1}, []int{2}, []int{3}, []int{0},
				},
			},
			0,
			1,
			4,
		},
	}

	for i, t_case := range test_cases {
		z_count, total_count := Sparsity(t_case.w, t_case.z)
		if z_count != t_case.z_count || total_count != t_case.total_count {
			t.Errorf("TestCase: %d: expected %d, %d, but got %d, %d.",
				i, t_case.z_count, t_case.total_count, z_count, total_count)
		}
	}
}
