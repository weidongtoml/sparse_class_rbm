// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"testing"
)

func Test_SelectKFromDist(t *testing.T) {
	test_cases := []struct {
		p      WeightT
		p_dist []WeightT
		k      int
	}{
		{
			0.01,
			[]WeightT{0.01, 0.02, 0.03, 0.04},
			0,
		}, {
			0.001,
			[]WeightT{0.01, 0.02, 0.03, 0.04},
			0,
		}, {
			0.02,
			[]WeightT{0.01, 0.02, 0.03, 0.04},
			1,
		}, {
			1.0,
			[]WeightT{0.01, 0.02, 0.03, 0.04},
			3,
		}, {
			0.9,
			[]WeightT{0.001, 0.01, 0.1, 0.2, 0.31, 0.379},
			5,
		}, {
			0.011,
			[]WeightT{0.001, 0.01, 0.1, 0.2, 0.31, 0.379},
			1,
		},
	}

	for i, t_case := range test_cases {
		k := SelectKFromDist(t_case.p, t_case.p_dist)
		if k != t_case.k {
			t.Errorf("TestCase #%d: expected %d but got %d.", i, t_case.k, k)
		}
	}
}
