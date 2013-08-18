// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"math/rand"
)

// Method sampleHGivenXY samples H according to the p.d. P(H|X, Y).
func (rbm *SparseClassRBM) sampleHGivenXY(h []WeightT, x []int, y int) {
	for i := range h {
		p := rbm.probOfHGivenXY(i, x, y)
		if RandomWeight() < p {
			h[i] = WeightT(1)
		} else {
			h[i] = WeightT(0)
		}
	}
}

// Method sampleXGivenH sample X according to the p.d. P(X|h).
func (rbm *SparseClassRBM) sampleXGivenH(x []int, h []WeightT) {
	for c := 0; c < rbm.x_class_num; c++ {
		p_dist := rbm.probOfXInClassCGivenH(c, h)
		x[c] = SampleKFromDistribution(p_dist)
	}
}

// Method sampleYGivenH samples Y according to the p.d. P(Y|H)
func (rbm *SparseClassRBM) sampleYGivenH(y *int, h []WeightT) {
	if RandomWeight() < Sigmoid(rbm.d+DotProduct(rbm.u, h)) {
		*y = 1
	} else {
		*y = 0
	}
}

// Method SampleKFromDistribution selects a sample from the multinomial distribution p_dist.
func SampleKFromDistribution(p_dist []WeightT) int {
	return SelectKFromDist(RandomWeight(), p_dist)
}

func SelectKFromDist(p WeightT, p_dist []WeightT) int {
	k := 0
	acc_prob := p_dist[0]
	for ; k < len(p_dist)-1; k++ {
		if p > acc_prob {
			acc_prob += p_dist[k+1]
		} else {
			break
		}
	}
	return k
}

func RandomWeight() WeightT {
	return WeightT(rand.Float64())
}
