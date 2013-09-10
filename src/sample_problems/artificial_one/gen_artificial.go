// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generates a set artificial data from testing the learning of RBM.
//

package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"rbm"
	"strings"
	"time"
)

type SimpleModel struct {
	feature_distribution [][]float64
	feature_weights      [][]float64
	threshold            float64
	noise_std            float64
}

func (m *SimpleModel) GenerateInstance() ([]string, string) {
	var x []string
	w := float64(0)
	for i, v := range m.feature_distribution {
		k := SelectKFromDist(rand.Float64(), v)
		x = append(x, fmt.Sprintf("%d:%d", i, k))
		w += m.feature_weights[i][k]
	}
	w += rand.NormFloat64() * m.noise_std
	y := "0"
	if Logit(w) < m.threshold {
		y = "1"
	}
	return x, y
}

// Selects one out of len(p_dist) based on the probability distribution of
// p_dist and the random value p.
func SelectKFromDist(p float64, p_dist []float64) int {
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

func Logit(x float64) float64 {
	x_exp := math.Exp(x)
	return x_exp / (1 + x_exp)
}

func GenTrainData() {
	rand.Seed(time.Now().Unix())

	model := SimpleModel{
		[][]float64{
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
		[][]float64{
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
		0.01,
	}

	instance_cnt := []int{5000, 1000, 2000}
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
			out_fd.WriteString(fmt.Sprintf("%s\t%s\n", y, strings.Join(x, "\t")))
		}
	}
}

func TrainRBM() {
	train_file := "./data_0.dat"
	validation_file := "./data_1.dat"

	var rbm_m rbm.SparseClassRBM
	(&rbm_m).Initialize(class_sizes, class_biases, hidden_layer_size, y_bias)

	train_data_accessor := rbm.NewInstanceLoader(train_file, len(class_sizes))
	defer train_data_accessor.Close()

	validation_data_accessor := rbm.NewInstanceLoader(validation_file, len(class_sizes))
	defer validation_data_accessor.Close()

	class_sizes := []int{4, 2, 5, 1}
	hidden_layer_size := 4

	class_biases, y_bias := rbm.GetBiases(class_sizes, train_data_accessor)

	var trainer rbm.RBMTrainer
	trainer.Initialize(&rbm_m, train_data_accessor, validation_data_accessor,
		0.01, 0.2, 0.9, 0.3, 1)

	trainer.Train()

	//Evaluate RBM
	//	test_file := "./data_2.dat"
}

func main() {
	gen_data := false
	if gen_data {
		GenTrainData()
	}

	TrainRBM()
}
