// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Collection of mathematical functions.

package rbm

import (
	"fmt"
	"math"
)

// Function Sigmoid returns the value of exp(x)/(1+exp(x)).
func Sigmoid(x WeightT) WeightT {
	p := math.Exp(float64(x))
	return WeightT(p / (1 + p))
}

// Function SoftPlus returns the value of log(1+exp(x)).
func SoftPlus(x WeightT) WeightT {
	return WeightT(math.Log(1.0 + math.Exp(float64(x))))
}

// Function Exp returns the value of e^x.
func Exp(x WeightT) WeightT {
	return WeightT(math.Exp(float64(x)))
}

// Function DotProduct calcuates the dot product of the given 2 arrays.
func DotProduct(a, b []WeightT) WeightT {
	if len(a) != len(b) {
		panic(fmt.Sprintf("Vector dimensions not the same: %d %d", len(a), len(b)))
	}
	p := WeightT(0)
	for i := range a {
		p += a[i] * b[i]
	}
	return p
}

// ScalarProduct calculates the scalar product of a multiply by b.
func ScalarProduct(a []WeightT, b WeightT) {
	for i, _ := range a {
		a[i] *= b
	}
}
