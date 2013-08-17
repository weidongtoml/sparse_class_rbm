// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	_ "math"
	_ "math/rand"
)

func (trainer *RBMTrainer) Initialize(rbm *SparseClassRBM,
	train_data_accessor DataInstanceAccessor,
	validation_data_accessor DataInstanceAccessor,
	learning_rate float32, regularization_rate float32,
	momentum_rate float32, gen_learn_importance float32, gibbs_chain_length uint32) {
	trainer.parameters = trainParameters{
		learning_rate,
		regularization_rate,
		momentum_rate,
		gen_learn_importance,
		gibbs_chain_length,
	}
	trainer.rbm = rbm
	trainer.training_data_accessor = train_data_accessor
	trainer.validation_data_accessor = validation_data_accessor
}

// Method DoTraining does single instance training.
func (trainer *RBMTrainer) DoTraining() {
	param := &trainer.parameters
	alpha := WeightT(param.gen_learn_importance)
	one_plus_alpha := WeightT(1 + param.gen_learn_importance)
	rbm := trainer.rbm
	data_instance := trainer.training_data_accessor.NextInstance()

	x := data_instance.x
	y := data_instance.y

	// CD-k
	x_hat := make([]int, rbm.x_class_num)
	y_hat := 0
	h_hat := make([]WeightT, rbm.h_num)
	if param.gen_learn_importance > 0 {
		copy(x_hat, x)
		y_hat = y
		for t := uint32(0); t < param.gibbs_chain_length; t++ {
			rbm.sampleHGivenXY(h_hat, x_hat, y_hat)
			rbm.sampleXGivenH(x_hat, h_hat)
			rbm.sampleYGivenH(&y_hat, h_hat)
		}
		// Final H uses probability, not samples.
		rbm.probDistOfHGivenXY(h_hat, x_hat, y_hat)
	}
	p_dist_h_given_x_posy := make([]WeightT, rbm.h_num)
	rbm.probDistOfHGivenXY(p_dist_h_given_x_posy, x, 1)

	p_dist_h_given_x_negy := make([]WeightT, rbm.h_num)
	rbm.probDistOfHGivenXY(p_dist_h_given_x_negy, x, 1)

	var p_dist_h_given_xy []WeightT
	if y == 1 {
		p_dist_h_given_xy = p_dist_h_given_x_posy
	} else {
		p_dist_h_given_xy = p_dist_h_given_x_negy
	}

	p_y_given_x := rbm.probOfYGivenX(x)

	//delta_W[c][j][k], valid only if X_c = k
	for j := 0; j < rbm.h_num; j++ {
		ep_hj_yx := p_dist_h_given_x_posy[j]*p_y_given_x + p_dist_h_given_x_negy[j]*(1-p_y_given_x)
		delta_c_j := one_plus_alpha*p_dist_h_given_xy[j] - ep_hj_yx - alpha*h_hat[j]
		delta_u_j := one_plus_alpha*p_dist_h_given_xy[j]*WeightT(y) - p_y_given_x*p_dist_h_given_x_posy[j] - alpha*h_hat[j]*WeightT(y_hat)
		for c, k := range x {
			delta_w_c_j_k := delta_c_j
			//when X_c != k, delta_w_c_j_k = 0
		}
	}
	for c, k := range x {
		delta_b_c_k := alpha * WeightT(k-x_hat[c])
	}
	delta_d := one_plus_alpha*WeightT(y) - p_y_given_x - alpha*WeightT(y_hat)
}
