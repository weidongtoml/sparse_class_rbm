package rbm

import (
	"math"
	"math/rand"
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

func (trainer *RBMTrainer) DoTraining() {
	param := &trainer.parameters
	rbm := trainer.rbm
	//data_instance := trainer.training_data_accessor.NextInstance()
	
	// CD-k
	if param.gen_learn_importance > 0 {
		x := make([]uint32, rbm.x_class_num)
		y := float64(0)
		h := make([]bool, rbm.h_num)
	
		for t := uint32(0); t < param.gibbs_chain_length; t++ {
			//Sample hidden units
			for j, _ := range h {
				energy := float64(rbm.c[j]) + y * float64(rbm.u[j])
				for c, _ := range rbm.w {
					for v := range x {
						energy += float64(rbm.w[c][j][v])
					}
				}
				p_hj_given_xy := sigmoid(energy)
				if rand.Float64() < p_hj_given_xy {
					h[j] = false
				} else {
					h[j] = true
				}
			}
			//Sample x
			for i, _ := range x {
				exp_sum_h_w := make(float64[], len(rbm.w[i]))
				for 
			}
		}
	}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}