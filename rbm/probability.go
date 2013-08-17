// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	_ "math"
)

// Method probDistOfHGivenXY calculates the probability distribution of
// p(h = [1]| X, Y) and store the result in h.
func (rbm *SparseClassRBM) probDistOfHGivenXY(h []WeightT, x []int, y int) {
	for i := range h {
		h[i] = rbm.probOfHGivenXY(i, x, y)
	}
}

// Method probOfHGivenXY calculates the probability of P(h_j = 1 | X, Y) using
// 	P(h_j = 1 | X, Y) = sigmoid(sum{0<=c<=C}(W[c][j][X_c]) + c[j] + U_j . Y)
func (rbm *SparseClassRBM) probOfHGivenXY(j int, x []int, y int) WeightT {
	s := rbm.c[j] + WeightT(y)*rbm.u[j] + rbm.wHDotX(j, x)
	return Sigmoid(s)
}

// Method probOfXInClassCGivenH calculates the probability of P(X_c | h)
//	E(X_c = k, H) = exp(sum{0 <= j <= |H}(W[c][j][k] * H[j]))
//	P(X_c = k | H) = E(X_c = k, H) / ( sum{0 <= q < |X_c|}(E(X_c = q, H) )
func (rbm *SparseClassRBM) probOfXInClassCGivenH(c int, h []WeightT) []WeightT {
	p := make([]WeightT, rbm.x_class_sizes[c])
	var denominator WeightT
	for k := 0; k < rbm.x_class_sizes[c]; k++ {
		s := WeightT(0.0)
		for j := 0; j < rbm.h_num; j++ {
			s += rbm.w[c][j][k] * h[j]
		}
		p[k] = Exp(s)
		denominator += p[k]
	}
	for k := 0; k < rbm.x_class_sizes[c]; k++ {
		p[k] /= denominator
	}
	return p
}

// Method probOfYGivenX calculates the probability of P(Y=1 | X)
//	P(Y=1|X) = exp{d + sum{0<=j<|H|}(sotfplus( w[j].X + c[j] + u[j] )) } /
//		( exp{ d + sum{0<=j<|H|}(sotfplus( w[j].X + c[j] + u[j] )) } +
//			 exp{ sum{0<=j<|H|}(sotfplus( w[j].X + c[j] )) } )
func (rbm *SparseClassRBM) probOfYGivenX(x []int) WeightT {
	neg := WeightT(0)
	pos := WeightT(0)
	for j := 0; j < rbm.h_num; j++ {
		w_dot_x_add_c := rbm.wHDotX(j, x) + rbm.c[j]
		neg += SoftPlus(w_dot_x_add_c)
		pos += SoftPlus(w_dot_x_add_c + rbm.u[j])
	}
	return Exp(rbm.d+pos) / (Exp(rbm.d+pos) + Exp(neg))
}

// Method wClassCDotX calculates the dot product of W[j] . X
func (rbm *SparseClassRBM) wHDotX(j int, x []int) WeightT {
	p := WeightT(0)
	for c := 0; c < rbm.x_class_num; c++ {
		p += rbm.w[c][j][x[c]]
	}
	return p
}
