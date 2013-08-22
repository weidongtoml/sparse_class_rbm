// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"testing"
)

func Test_Initialize(t *testing.T) {
	num_classes := int(3)
	feature_classes := []int{1, 2, 3}
	member_biases := [][]WeightT{
		{0.1},
		{0.2, 0.3},
		{0.4, 0.05, -0.2}}
	number_hidden_units := int(4)
	positive_y_bias := WeightT(0.03)

	rbm := new(SparseClassRBM)
	rbm.Initialize(feature_classes, member_biases, number_hidden_units, positive_y_bias)

	if rbm.h_num != number_hidden_units {
		t.Errorf("Expected the number of hidden units to be %d but got %d\n",
			number_hidden_units, rbm.h_num)
	}

	if rbm.x_class_num != num_classes {
		t.Errorf("Expected the number of feature classes to be %d but got %d\n",
			num_classes, rbm.x_class_num)
	}

	for i, s := range feature_classes {
		if rbm.ClassSize(i) != s {
			t.Errorf("Expected class %d has size %d, but got %d.", i, s, rbm.ClassSize(i))
		}
	}

	//test for the dimenions of W
	if len(rbm.w) != num_classes {
		t.Errorf("Expected len(W) to be %d but got %d\n", num_classes, len(rbm.w))
	}

	for c, class_w := range rbm.w {
		if len(class_w) != number_hidden_units {
			t.Errorf("Expected dim(rbm.w[]) to be equal to %d but got %d\n",
				number_hidden_units, len(class_w))
		}
		for h, class_w_h := range class_w {
			if len(class_w_h) != feature_classes[c] {
				t.Errorf("Expected len(rbm.w[%d][%d]) to be %d but got %d\n",
					c, h, feature_classes[c], len(class_w_h))
			}
		}
	}

	//test for the visible bias
	if len(rbm.b) != num_classes {
		t.Errorf("Visible biases has number of classes %d, which is not equal to %d\n",
			len(rbm.b), num_classes)
	}

	for c, class_bias := range member_biases {
		for b, bias := range class_bias {
			if bias != rbm.b[c][b] {
				t.Errorf("Expected b[%d][%d] to be [%f] but got [%f] instead\n",
					c, b, bias, rbm.b[c][b])
			}
		}
	}

	//test for hidden unit bias
	if len(rbm.c) != number_hidden_units {
		t.Errorf("Expected len(rbm.c) to be %d but got %d\n",
			number_hidden_units, len(rbm.c))
	}

	//test for U
	if len(rbm.u) != number_hidden_units {
		t.Errorf("Expected len(len(rbm.u)) to be %d but got %d\n",
			number_hidden_units, len(rbm.u))
	}

	//test for d
	if rbm.d != positive_y_bias {
		t.Errorf("Expected rbm.d to be %f, but got %f\n",
			positive_y_bias, rbm.d)
	}

	empty_rbm := rbm.CloneEmpty()
	if empty_rbm.NumOfVisibleClasses() != rbm.NumOfVisibleClasses() {
		t.Errorf("Number of visible classes not equal, expected %d, but got %d.",
			rbm.NumOfVisibleClasses(), empty_rbm.NumOfVisibleClasses())
	} else {
		for i := 0; i < rbm.NumOfVisibleClasses(); i++ {
			if empty_rbm.ClassSize(i) != rbm.ClassSize(i) {
				t.Errorf("Class %d size not equal, expectd %d but got %d.",
					i, rbm.ClassSize(i), empty_rbm.ClassSize(i))
			}
		}
	}
	if empty_rbm.SizeOfHiddenLayer() != rbm.SizeOfHiddenLayer() {
		t.Errorf("Expected hiddeny layer size to be %d but got %d.",
			rbm.SizeOfHiddenLayer(), empty_rbm.SizeOfHiddenLayer())
	}
	if len(empty_rbm.w) != len(rbm.w) {
		t.Errorf("len(w) not equal")
	}
	for i, _ := range rbm.w {
		if len(empty_rbm.w[i]) != len(rbm.w[i]) {
			t.Errorf("len(w[%d]) not equal.", i)
		}
		for v, _ := range rbm.w[i] {
			if len(rbm.w[i][v]) != len(empty_rbm.w[i][v]) {
				t.Errorf("len(w[%d][%d]) not equal.", i, v)
			}
		}
	}
	if len(empty_rbm.b) != len(rbm.b) {
		t.Errorf("len(rbm.b) not equal.")
	}
	for i, _ := range rbm.b {
		if len(empty_rbm.b[i]) != len(rbm.b[i]) {
			t.Errorf("len(b[%d]) not equal, expected %d, but got %d.", i,
				len(rbm.b[i]), len(empty_rbm.b[i]))
		}
	}
	if len(empty_rbm.c) != len(rbm.c) {
		t.Errorf("len(rbm.c) not equal.")
	}
	if len(empty_rbm.u) != len(rbm.u) {
		t.Errorf("len(rbm.u) not equal.")
	}
}
