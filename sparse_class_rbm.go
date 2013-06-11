/**
 * SparseClassRBM:
 * E(y, X, h) = -SumOver_0<=i<C_{h^T . W(i) . e_x_i + b(i) . e_x_i} - h^T . c - d . y - h^T . U . y
 * p(y, X, h) = exp(-E(y,X,h))/Z
 */
package SparseClassRBM

import (
	"math"
)

type SparseClassRBM struct {
	w           [][]float32 //interactions between X and h
	b           []float32   //bias of X
	c           []float32   //bias of h
	u           []float32   //interactions between y and h
	d           float32     //bias of y
	x_class_num uint32      //number of Classes in X
	h_num       uint32      //number of hidden units
}

type DataInstance struct {
	x []uint32 //values of each classes, in the order of Class0, Class1, ...
	y float32  //values of Y
}

type TrainParameters struct {
	learning_rate        float32 //equivalent to the \theta in the Delta Rule
	regularization_rate  float32 //equivalent to the \lambda in the Delta Rule
	momentum_rate        float32 //equivalent to the \mu in the Delta Rule
	gen_learn_importance float32 //equivalent to the \alpha in the hybrid learning equation
}

func softplus(x *float32) {
	return math.Log(1 + x)
}

func (rbm *SparseClassRBM) GetPrediction(instance *DataInstance) float32 {
	sum_of_x_weights = 0.0
	for x_class, x_value := range instance.x {
		sum_of_x_weights += rbm.w[x_class][x_value]
	}
	energy_y_0, energy_y_1 = 0.0, rbm.d
	for c_index, c_value := range rbm.c {
		energy_y_0 += softplus(sum_of_x_weights + c_value)
		energy_y_1 += softplus(sum_of_x_weights + c_value + rbm.u[c_index])
	}
	return math.Exp(energy_y_1) / (math.Exp(energy_y_1) + math.Exp(energy_y_0))
}


