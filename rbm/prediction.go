// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

// Calculate P(y = 1|X)
// Note instance.y is ignored in the calcuation.
func (rbm *SparseClassRBM) GetPrediction(instance *DataInstance) WeightT {
	return rbm.probOfYGivenX(instance.x)
}
