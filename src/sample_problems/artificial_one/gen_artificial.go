// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generates a set artificial data from testing the learning of RBM.
//

package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"rbm"
	"strings"
	"time"
)

type SimpleModel struct {
	feature_distribution [][]rbm.WeightT
	feature_weights      [][]rbm.WeightT
	threshold            rbm.WeightT
	noise_std            rbm.WeightT
}

// GenerateInstance generates a data sample form the given model.
func (m *SimpleModel) GenerateInstance() ([]int, int) {
	x := make([]int, len(m.feature_distribution))
	for i, v := range m.feature_distribution {
		k := SelectKFromDist(rbm.WeightT(rand.Float64()), v)
		x[i] = k
	}
	y := 0
	if m.Predict(x, m.noise_std) > m.threshold {
		y = 1
	}

	return x, y
}

// InstanceToString converts the given instance to string.
func InstanceToString(x []int, y int) string {
	label := ""
	if y == 1 {
		label = "1\t0"
	} else {
		label = "0\t1"
	}

	var instance []string
	for i, v := range x {
		instance = append(instance, fmt.Sprintf("%d:%d", i, v))
	}
	return label + "\t" + strings.Join(instance, "\t")
}

// Predict makes a prediction with the given input.
func (m *SimpleModel) Predict(x []int, noise rbm.WeightT) rbm.WeightT {
	w := rbm.WeightT(0)
	for i, k := range x {
		w += m.feature_weights[i][k]
	}
	w += noise
	return Sigmoid(w)
}

func (m SimpleModel) GetPrediction(instance *rbm.DataInstance) rbm.WeightT {
	return m.Predict(instance.GetX(), 0)
}

// Selects one out of len(p_dist) based on the probability distribution of
// p_dist and the random value p.
func SelectKFromDist(p rbm.WeightT, p_dist []rbm.WeightT) int {
	k := 0
	acc_prob := p_dist[0]
	for ; k < len(p_dist)-1; k++ {
		if p > acc_prob {
			acc_prob += p_dist[k+1]
		} else {
			break
		}
	}
	return k
}

func Sigmoid(x rbm.WeightT) rbm.WeightT {
	x_exp := math.Exp(float64(x))
	return rbm.WeightT(x_exp / (1 + x_exp))
}

func GenTrainData() {
	rand.Seed(time.Now().Unix())

	model := SimpleModel{
		[][]rbm.WeightT{
			{
				0.1, 0.2, 0.3, 0.4,
			}, {
				0.5, 0.5,
			}, {
				0.7, 0.01, 0.09, 0.1, 0.1,
			}, {
				1.0,
			},
		},
		[][]rbm.WeightT{
			{
				0.2, -0.3, 1.2, 3.5,
			}, {
				0.2, 0.3,
			}, {
				-0.01, 0.39, 0.2, -1.0, 0.09,
			}, {
				1.0,
			},
		},
		0.8,
		0.0,
	}

	instance_cnt := []int{100000, 20000, 40000}
	prefix := "./data"

	for i, c := range instance_cnt {
		out_fd, err := os.Create(fmt.Sprintf("%s_%d.dat", prefix, i))
		if err != nil {
			fmt.Printf("Failed to create file: %s, %s.\n", out_fd, err)
			return
		}
		defer out_fd.Close()
		for j := 0; j < c; j++ {
			x, y := model.GenerateInstance()
			out_fd.WriteString(InstanceToString(x, y) + "\n")
		}
	}

	validation_file := "./data_1.dat"
	validation_data_accessor := rbm.NewInstanceLoader(validation_file, len(model.feature_distribution))
	defer validation_data_accessor.Close()

	validation_auc := rbm.ROCAuc(model, validation_data_accessor)
	log_likelihood := rbm.LogLikelihood(model, validation_data_accessor)
	rmse := rbm.RMSE(model, validation_data_accessor)
	fmt.Printf("Max Validation AUC: %f\n", validation_auc)
	fmt.Printf("Max Validation LogLikelihood: %f\n", log_likelihood)
	fmt.Printf("max RMSE: %f\n", rmse)
}

func TrainRBM() {
	train_file := "./data_0.dat"
	validation_file := "./data_1.dat"
	class_sizes := []int{4, 2, 5, 1}
	hidden_layer_size := 2

	train_data_accessor := rbm.NewInstanceLoader(train_file, len(class_sizes))
	defer train_data_accessor.Close()

	validation_data_accessor := rbm.NewInstanceLoader(validation_file, len(class_sizes))
	defer validation_data_accessor.Close()

	class_biases, y_bias := rbm.GetBiases(class_sizes, train_data_accessor)
	var rbm_m rbm.SparseClassRBM
	(&rbm_m).Initialize(class_sizes, class_biases, hidden_layer_size, y_bias)

	var trainer rbm.RBMTrainer

	learning_rate := rbm.WeightT(0.001)
	regularization := rbm.WeightT(0.0)
	momentum := rbm.WeightT(0.0)
	gen_learning_imp := rbm.WeightT(0.0)
	gibs_chain_len := 1

	trainer.Initialize(&rbm_m, train_data_accessor, validation_data_accessor,
		learning_rate, regularization, momentum, gen_learning_imp, gibs_chain_len)

	trainer.Train()

	//Evaluate RBM
	//	test_file := "./data_2.dat"
}

func main() {
	gen_data := true
	if gen_data {
		GenTrainData()
	}

	TrainRBM()
}
