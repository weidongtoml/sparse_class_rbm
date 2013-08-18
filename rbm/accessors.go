// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

func (rbm *SparseClassRBM) SizeOfHiddenLayer() int {
	return (*rbm).h_num
}

func (rbm *SparseClassRBM) NumOfVisibleClasses() int {
	return (*rbm).x_class_num
}

func (rbm *SparseClassRBM) ClassSize(class_id int) int {
	return (*rbm).x_class_sizes[class_id]
}

// Method getW returns the interaction matrix of X and H
func (rbm *SparseClassRBM) W(h_index, c_index, c_value int) WeightT {
	return (*rbm).w[c_index][h_index][c_value]
}

func (rbm *SparseClassRBM) SetW(h_index, c_index, c_value int, v WeightT) {
	(*rbm).w[c_index][h_index][c_value] = v
}

// Method getB returns the pointer to the bias of X
func (rbm *SparseClassRBM) B(c_index, c_value int) WeightT {
	return (*rbm).b[c_index][c_value]
}

func (rbm *SparseClassRBM) SetB(c_index, c_value int, v WeightT) {
	(*rbm).b[c_index][c_value] = v
}

// Method getC returns the pointer to the bias of H
func (rbm *SparseClassRBM) C(h_index int) WeightT {
	return (*rbm).c[h_index]
}

func (rbm *SparseClassRBM) SetC(h_index int, v WeightT) {
	(*rbm).c[h_index] = v
}

// Method getU returns the pointer to the interaction vector of Y and H
func (rbm *SparseClassRBM) U(h_index int) WeightT {
	return (*rbm).u[h_index]
}

func (rbm *SparseClassRBM) SetU(h_index int, v WeightT) {
	(*rbm).u[h_index] = v
}

// Method getD returns the pointer to the bias of Y
func (rbm *SparseClassRBM) D() WeightT {
	return (*rbm).d
}

func (rbm *SparseClassRBM) SetD(v WeightT) {
	(*rbm).d = v
}
