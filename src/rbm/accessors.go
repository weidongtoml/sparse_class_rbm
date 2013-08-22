// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

// Method SizeOfHiddenLayer returns the number of hidden units in the model.
func (rbm *SparseClassRBM) SizeOfHiddenLayer() int {
	return (*rbm).h_num
}

// Method NumOfVisibleClasses returns the number of visible classes.
func (rbm *SparseClassRBM) NumOfVisibleClasses() int {
	return (*rbm).x_class_num
}

// Method ClassSize returns the size of the given visible unit class.
func (rbm *SparseClassRBM) ClassSize(class_id int) int {
	return (*rbm).x_class_sizes[class_id]
}

// Method W returns the interaction of X and H
func (rbm *SparseClassRBM) W(h_index, c_index, c_value int) WeightT {
	return (*rbm).w[c_index][h_index][c_value]
}

// Method W sets the interaction of X and H
func (rbm *SparseClassRBM) SetW(h_index, c_index, c_value int, v WeightT) {
	(*rbm).w[c_index][h_index][c_value] = v
}

// Method B returns the bias of X.
func (rbm *SparseClassRBM) B(c_index, c_value int) WeightT {
	return (*rbm).b[c_index][c_value]
}

// Method SetB sets the bias of X.
func (rbm *SparseClassRBM) SetB(c_index, c_value int, v WeightT) {
	(*rbm).b[c_index][c_value] = v
}

// Method C returns the bias of H.
func (rbm *SparseClassRBM) C(h_index int) WeightT {
	return (*rbm).c[h_index]
}

// Method C sets the bias of H.
func (rbm *SparseClassRBM) SetC(h_index int, v WeightT) {
	(*rbm).c[h_index] = v
}

// Method U returns the interaction of Y and H.
func (rbm *SparseClassRBM) U(h_index int) WeightT {
	return (*rbm).u[h_index]
}

// Method U sets the interaction of Y and H
func (rbm *SparseClassRBM) SetU(h_index int, v WeightT) {
	(*rbm).u[h_index] = v
}

func (rbm *SparseClassRBM) UVector() []WeightT {
	return (*rbm).u
}

// Method D returns the the bias of Y
func (rbm *SparseClassRBM) D() WeightT {
	return (*rbm).d
}

// Method SetD sets the the bias of Y
func (rbm *SparseClassRBM) SetD(v WeightT) {
	(*rbm).d = v
}
