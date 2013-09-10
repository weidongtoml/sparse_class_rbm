// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"fmt"
)

const (
	KMinDeltaAUC = 0.001 //minimum AUC change to justify another round of training
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

// Train an RBM.
func (trainer *RBMTrainer) Train() {
	pos_delta := trainer.rbm.NewDeltaT()
	neg_delta := trainer.rbm.NewDeltaT()
	prev_auc := float64(0)
	epoch := 0
	for {
		has_pos, has_neg := trainer.doGradient(pos_delta, neg_delta)
		if has_pos {
			trainer.updateModel(pos_delta)
		}
		if has_neg {
			trainer.updateModel(neg_delta)
		}
		if !has_pos && !has_neg {
			auc := ROCAuc(trainer.rbm, trainer.validation_data_accessor)

			//TODO(weidoliang): output various model statistics.
			fmt.Printf("Epoch: %d\n", epoch)
			fmt.Printf("AUC: %f\n", auc)
			trainer.ModelStats()
			if auc-prev_auc < KMinDeltaAUC {
				break
			} else {
				trainer.training_data_accessor.Reset()
			}
			epoch++
		}
	}
}

func (trainer *RBMTrainer) ModelStats() {
	w_sparsity := trainer.rbm.SparsityOfW()
	u_sparsity := trainer.rbm.SparsityOfU()
	fmt.Printf("Sparsity: \nW: %f\nU: %f\n", w_sparsity, u_sparsity)
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
