// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	_ "fmt"
	"math"
	"testing"
)

func EqualWithinPrecision(a, b, p float64) bool {
	return math.Abs(a-b) <= p
}

const kPrecision = 0.000001

func Test_Sigmoid(t *testing.T) {
	if !EqualWithinPrecision(0.5, Sigmoid(0), kPrecision) {
		t.Errorf("Expected logit(0) to be 0.5 but got %v", Sigmoid(0))
	}
	if !EqualWithinPrecision(0.731058, Sigmoid(1), kPrecision) {
		t.Errorf("Expected logit(1) to be 0.731058, but got %v", Sigmoid(1))
	}
	if !EqualWithinPrecision(0.9999546, Sigmoid(10), kPrecision) {
		t.Errorf("Expected logit(10) to be 0.99995, but got %v", Sigmoid(10))
	}

}
