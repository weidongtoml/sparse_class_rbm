package rbm

import (
	"fmt"
	"math/rand"
)

/**
 * Initialize a SparseClassRBM
 *
 * @see Ref. [1], 8. The initial values of the weights and bias.
 *     "The weigths are typically initialized to small random values chosen from
 *	a zero-mean Gaussian with a standard deviation of about 0.01..."
 *     "...initialize the bias of visible unit i to log[p_i/(1-p_i)] where p_i
 *  is the proportion of training vectors in which unit i is on..."
 *     "...using a sparsity target probability of t (see section 11), it makes
 *  sense to initialize the hidden biases to be log[t/(1-t)]. Otherwise, initial
 *  hidden biases of 0 are usually fine. It is also possible to start the hidden
 *  units with quite large negative biases of about -4 as a crude way of 
 *  encouraging sparsity..." 
 */
func (rbm *SparseClassRBM) Initialize(feature_classes []uint32,
	member_biases [][]float32,
	num_hidden_units uint32,
	positive_y_bias float32) {
	rbm.x_class_num = uint32(len(feature_classes))
	rbm.h_num = num_hidden_units

	if num_hidden_units < 1 {
		panic("Number of hidden units must be greater than 0.")
	}
	if len(feature_classes) != len(member_biases) {
		panic(fmt.Sprintf("Number of classes specified by feature_classes[%d] is not equal to that of member_biases[%d]",
			len(feature_classes), len(member_biases)))
	}
	for c, k := range feature_classes {
		if uint32(len(member_biases[c])) != k {
			panic(fmt.Sprintf("Bias and class members #%d do not match [%d != %d]", c,
				len(member_biases[c]), k))
		}
	}

	rbm.init_W(num_hidden_units, feature_classes)
	rbm.init_B(feature_classes, member_biases)
	rbm.init_C(num_hidden_units)
	rbm.init_U_D(num_hidden_units, positive_y_bias)
}

//Initialize W with Norm(0, 0.01)
func (rbm *SparseClassRBM) init_W(num_hidden_units uint32, feature_classes []uint32) {
	const w_deviation float32 = 0.01
	rbm.w = make([][][]float32, len(feature_classes))
	for c, k := range feature_classes {
		rbm.w[c] = make([][]float32, num_hidden_units)
		for h, _ := range rbm.w[c] {
			rbm.w[c][h] = make([]float32, k)
			for v, _ := range rbm.w[c][h] {
				rbm.w[c][h][v] = zero_mean_norm_rand(w_deviation)
			}
		}
	}
}

//Initialize visible biases
func (rbm *SparseClassRBM) init_B(feature_classes []uint32, member_biases [][]float32) {
	rbm.b = make([][]float32, len(member_biases))
	for c, k := range feature_classes {
		rbm.b[c] = make([]float32, k)
		for v, _ := range rbm.b[c] {
			rbm.b[c][v] = member_biases[c][v]
		}
	}
}

//Initialize the hidden biases
func (rbm *SparseClassRBM) init_C(num_hidden_units uint32) {
	const default_hidden_bias float32 = -4
	rbm.c = make([]float32, num_hidden_units)
	for h, _ := range rbm.c {
		rbm.c[h] = default_hidden_bias
	}
}

//Initialize U with Norm(0, 0.01)
func (rbm *SparseClassRBM) init_U_D(num_hidden_units uint32, positive_y_bias float32) {
	const u_devivation float32 = 0.01
	rbm.u = make([]float32, num_hidden_units)
	for y, _ := range rbm.u {
		rbm.u[y] = zero_mean_norm_rand(u_devivation)
	}
	rbm.d = positive_y_bias
}

func zero_mean_norm_rand(stddev float32) float32 {
	return float32(rand.NormFloat64()) * stddev
}
