// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"io"
	"log"
)

type deltaWT struct {
	h_index int
	c_index int
	c_value int
	delta_v WeightT
}

type deltaBT struct {
	c_index int
	c_value int
	delta_v WeightT
}

type deltaT struct {
	delta_w []deltaWT
	delta_b []deltaBT
	delta_d WeightT
	delta_c []WeightT
	delta_u []WeightT
}

func (d *deltaT) ScalarProduct(x int) {
	if x == 1 {
		return
	}
	x_t := WeightT(x)
	for i, _ := range d.delta_w {
		d.delta_w[i].delta_v *= x_t
	}
	for i, _ := range d.delta_b {
		d.delta_b[i].delta_v *= x_t
	}
	d.delta_d *= x_t
	ScalarProduct(d.delta_c, x_t)
	ScalarProduct(d.delta_u, x_t)
}

func (delta *deltaT) Clear() {
	(*delta).delta_w = (*delta).delta_w[0:0]
	(*delta).delta_b = (*delta).delta_b[0:0]
}

func (rbm *SparseClassRBM) NewDeltaT() *deltaT {
	var delta deltaT
	delta.delta_c = make([]WeightT, rbm.SizeOfHiddenLayer())
	delta.delta_u = make([]WeightT, rbm.SizeOfHiddenLayer())
	return &delta
}

func (trainer *RBMTrainer) doGradient(pos_delta, neg_delta *deltaT) (bool, bool) {
	has_pos, has_neg := false, false
	var data_instance DataInstance
	for {
		instance, err := trainer.training_data_accessor.NextInstance()
		if err == io.EOF {
			return has_pos, has_neg
		}
		if trainer.rbm.IsValidInput(instance) {
			data_instance = instance
			break
		} else {
			log.Printf("invalid input: %v.\n", instance)
		}
	}

	pos_delta.Clear()
	neg_delta.Clear()
	if data_instance.pos_y > 0 {
		has_pos = true
		trainer.doInstanceGradient(data_instance.x, 1, pos_delta)
		pos_delta.ScalarProduct(data_instance.pos_y)
	}
	if data_instance.neg_y > 0 {
		has_neg = true
		trainer.doInstanceGradient(data_instance.x, 0, neg_delta)
		pos_delta.ScalarProduct(data_instance.neg_y)
	}
	return has_pos, has_neg
}

func (trainer *RBMTrainer) doInstanceGradient(x []int, y int, delta *deltaT) {
	rbm := trainer.rbm
	param := &trainer.parameters
	alpha := WeightT(param.gen_learn_importance)
	one_plus_alpha := WeightT(1 + param.gen_learn_importance)

	delta.Clear()
	// CD-k
	x_hat := make([]int, rbm.NumOfVisibleClasses())
	y_hat := 0
	h_hat := make([]WeightT, rbm.SizeOfHiddenLayer())
	if param.gen_learn_importance > 0 {
		copy(x_hat, x)
		y_hat = y
		for t := 0; t < param.gibbs_chain_length; t++ {
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

	//Gradient Calculation
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
		(*delta).delta_c[j] = delta_c_j
		(*delta).delta_u[j] = one_plus_alpha*p_dist_h_given_xy[j]*WeightT(y) - p_y_given_x*p_dist_h_given_x_posy[j] - alpha*h_hat[j]*WeightT(y_hat)
		for c, k := range x {
			delta_w_c_j_k := delta_c_j
			//when X_c != k, delta_w_c_j_k = 0
			(*delta).delta_w = append((*delta).delta_w, deltaWT{j, c, k, delta_w_c_j_k})
		}
	}
	for c, k := range x {
		(*delta).delta_b = append((*delta).delta_b, deltaBT{c, k, alpha * WeightT(k-x_hat[c])})
	}
	(*delta).delta_d = one_plus_alpha*WeightT(y) - p_y_given_x - alpha*WeightT(y_hat)
}
