// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

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
		//delta_theta := eta*delta_v - lambda*cur_theta + mu*prev_delta_theta
		delta_theta := eta*delta_v + mu*prev_delta_theta
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
		//delta_theta := eta*delta.delta_d - lambda*cur_theta + mu*prev_delta_theta
		delta_theta := eta*delta.delta_d + mu*prev_delta_theta
		rbm.SetD(cur_theta + delta_theta)

		prev_delta.SetD(delta_theta)
	}
}
