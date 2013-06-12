package rbm

import (
	"testing"
)

func Test_Initialize(t *testing.T) {
	num_classes := uint32(3)
	feature_classes := []uint32{1, 2, 3}
	member_biases := [][]float32{
		[]float32{0.1},
		[]float32{0.2, 0.3},
		[]float32{0.4, 0.05, -0.2}}
	number_hidden_units := uint32(4)
	positive_y_bias := float32(0.03)

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

	if uint32(len(rbm.b)) != num_classes {
		t.Errorf("Visible biases has number of classes %d, which is not equal to %d\n",
			len(rbm.b), num_classes)
	}

}
