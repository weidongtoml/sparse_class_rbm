// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rbm provides and implementation of the SparseClassRBM for
// classification/regression.
//
// SparseClassRBM:
//  E(y, X, h) = -SumOver_0<=i<C_{h^T . W(i) . e_x_i + b(i) . e_x_i} - h^T . c - d . y - h^T . U . y
//  p(y, X, h) = exp(-E(y,X,h))/Z
//
// Reference:
//  [1]. Hinton, 2010, A Practical Guide to Training Restricted Boltzmann Machines (Ver. 1)
//
// TODO:
//  One visible bias per feature class VS one visible bias per visible units
//
package rbm
