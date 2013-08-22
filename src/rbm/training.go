// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

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
