// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"bufio"
	"common/util"
	"fmt"
	"os"
	"testing"
)

func saveDataToFile(filename string, data []DataInstance) error {
	return util.WithNewOpenFileAsBufioWriter(filename,
		func(w *bufio.Writer) error {
			for _, d := range data {
				fmt.Fprintf(w, "%d\t%d", d.pos_y, d.neg_y)
				for i, v := range d.x {
					fmt.Fprintf(w, "\t%d:%d", i, v)
				}
				fmt.Fprint(w, "\n")
			}
			return nil
		})
}

func Test_Training(t *testing.T) {
	train_file := "./training.txt"
	test_file := "./test.txt"
	class_sizes := []int{2, 3, 4, 5}
	class_biases := [][]WeightT{
		{0.01, 0.02},
		{0.03, 0.04, 0.05},
		{-0.01, -0.02, -0.03, -0.04},
		{0.01, -0.02, -0.03, 0.04, 1.2},
	}
	y_bias := WeightT(0.3)
	hidden_layer_size := 4

	//TODO(weidoliang): automatically generates Xs from a couple of mathematical
	// functions and generates Y based on conjunctive-disjunction of the values of Xs.
	train_data := []DataInstance{
		{
			[]int{0, 1, 1, 1},
			2,
			1,
		}, {
			[]int{1, 2, 1, 0},
			0,
			1,
		}, {
			[]int{1, 0, 3, 4},
			1,
			0,
		}, {
			[]int{1, 2, 3, 2},
			10,
			2,
		},
	}

	saveDataToFile(train_file, train_data)
	defer os.Remove(train_file)

	saveDataToFile(test_file, train_data)
	defer os.Remove(test_file)

	var rbm SparseClassRBM
	(&rbm).Initialize(class_sizes, class_biases, hidden_layer_size, y_bias)

	train_data_accessor := NewInstanceLoader(train_file, len(class_sizes))
	defer train_data_accessor.Close()

	test_data_accessor := NewInstanceLoader(test_file, len(class_sizes))
	defer test_data_accessor.Close()

	var trainer RBMTrainer
	trainer.Initialize(&rbm, train_data_accessor, test_data_accessor,
		0.01, 0.2, 0.9, 0.3, 1)

	trainer.Train()

	fmt.Printf("Training done.\n")
}
