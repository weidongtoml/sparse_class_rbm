// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"fmt"
	"math"
)

const kPrecision = 0.000001

func EqualWithinPrecesionF64(a, b, p float64) bool {
	return math.Abs(a-b) <= p
}

func EqualWithinPrecision(a, b WeightT, p float64) bool {
	return math.Abs(float64(a-b)) <= p
}

func ArraysEqualWithinPrecision(a, b []WeightT, p float64) bool {
	if len(a) != len(b) {
		panic(fmt.Sprintf("Array sizes not equal: %v, %v.", a, b))
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}
