// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"io"
	"log"
)

// Initialize a trainer.
func (trainer *RBMTrainer) Initialize(rbm *SparseClassRBM,
	train_data_accessor DataInstanceAccessor,
	validation_data_accessor DataInstanceAccessor,
	learning_rate WeightT, regularization_rate WeightT,
	momentum_rate WeightT, gen_learn_importance WeightT, gibbs_chain_length int) {
	trainer.parameters = trainParameters{
		learning_rate,
		regularization_rate,
		momentum_rate,
		gen_learn_importance,
		gibbs_chain_length,
	}
	trainer.rbm = rbm
	trainer.prev_delta = rbm.CloneEmpty()
	trainer.training_data_accessor = train_data_accessor
	trainer.validation_data_accessor = validation_data_accessor
}

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

func (rbm *SparseClassRBM) NewDeltaT() *deltaT {
	var delta deltaT
	delta.delta_c = make([]WeightT, rbm.SizeOfHiddenLayer())
	delta.delta_u = make([]WeightT, rbm.SizeOfHiddenLayer())
	return &delta
}

func (delta *deltaT) Clear() {
	(*delta).delta_w = (*delta).delta_w[0:0]
	(*delta).delta_b = (*delta).delta_b[0:0]
}

func (trainer *RBMTrainer) Train() {
	pos_delta := trainer.rbm.NewDeltaT()
	neg_delta := trainer.rbm.NewDeltaT()
	for {
		has_pos, has_neg := trainer.doGradient(pos_delta, neg_delta)
		if has_pos {
			trainer.updateModel(pos_delta)
		}
		if has_neg {
			trainer.updateModel(neg_delta)
		}
		if !has_pos && !has_neg {
			//TODO(weidoliang): evaluate model result and only when obtain the best
			//resilt do we stop training.
			break
		}
	}
}

func (rbm *SparseClassRBM) IsValidInput(instance DataInstance) bool {
	if (instance.pos_y < 0 || instance.neg_y < 0) ||
		(instance.pos_y == 0 && instance.neg_y == 0) {
		return false
	}
	for i := 0; i < rbm.NumOfVisibleClasses(); i++ {
		if instance.x[i] >= rbm.ClassSize(i) {
			return false
		}
	}
	return true
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

func (trainer *RBMTrainer) updateModel(delta *deltaT) {
	rbm := trainer.rbm
	prev_delta := trainer.prev_delta
	eta := trainer.parameters.learning_rate
	lambda := trainer.parameters.regularization_rate
	mu := trainer.parameters.momentum_rate

	//Update interaction matrix of X and H
	for _, d := range delta.delta_w {
		prev_delta_theta := prev_delta.W(d.h_index, d.c_index, d.c_value)

		cur_theta := rbm.W(d.h_index, d.c_index, d.c_value)
		delta_theta := eta*d.delta_v - lambda*cur_theta + mu*prev_delta_theta
		rbm.SetW(d.h_index, d.c_index, d.c_value, cur_theta+delta_theta)

		prev_delta.SetW(d.h_index, d.c_index, d.c_value, delta_theta)
	}

	//Update interaction bias of X
	for _, d := range delta.delta_b {
		prev_delta_theta := prev_delta.B(d.c_index, d.c_value)

		cur_theta := rbm.B(d.c_index, d.c_value)
		delta_theta := eta*d.delta_v - lambda*cur_theta + mu*prev_delta_theta
		rbm.SetB(d.c_index, d.c_value, cur_theta+delta_theta)

		prev_delta.SetB(d.c_index, d.c_value, delta_theta)
	}

	//Update bias of H
	for j, delta_v := range delta.delta_c {
		prev_delta_theta := prev_delta.C(j)

		cur_theta := rbm.C(j)
		delta_theta := eta*delta_v - lambda*cur_theta + mu*prev_delta_theta
		rbm.SetC(j, cur_theta+delta_theta)

		prev_delta.SetC(j, delta_theta)
	}

	//Update interactions between Y and H
	for j, delta_v := range delta.delta_u {
		prev_delta_theta := prev_delta.U(j)

		cur_theta := rbm.U(j)
		delta_theta := eta*delta_v - lambda*cur_theta + mu*prev_delta_theta
		rbm.SetU(j, cur_theta+delta_theta)

		prev_delta.SetU(j, delta_theta)
	}

	//Update bias of Y
	{
		prev_delta_theta := prev_delta.D()

		cur_theta := rbm.D()
		delta_theta := eta*delta.delta_d - lambda*cur_theta + mu*prev_delta_theta
		rbm.SetD(cur_theta + delta_theta)

		prev_delta.SetD(delta_theta)
	}
}
