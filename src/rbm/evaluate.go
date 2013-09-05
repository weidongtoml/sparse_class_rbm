// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A collection of functions and methods for evaluating the given model.

package rbm

import (
	"fmt"
	"io"
	"math"
	"reflect"
	"sort"
)

type BinaryClassifier interface {
	GetPrediction(instance *DataInstance) WeightT
}

type Coordinate struct {
	n_pos int
	n_neg int
	p     WeightT
}

type Coordinates []*Coordinate

func (c Coordinates) Len() int {
	return len(c)
}

func (c Coordinates) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}

func (c Coordinates) Less(i, j int) bool {
	return c[i].p < c[j].p
}

// ROCAuc calcuates the area under the receiver operating curve.
func ROCAuc(classifier BinaryClassifier, data_accessor *DataInstanceAccessor) float64 {
	coordinates := ROC(classifier, data_accessor)

	return 0
}

// ROC returns the coordinate of the Receiver Operating Curve based based on the
// given
func ROC(classifier BinaryClassifier, data_accessor DataInstanceAccessor) Coordinates {
	data_accessor.Reset()
	eval_result := make(map[WeightT]*Coordinate)
	for {
		instance, err := data_accessor.NextInstance()
		if err == io.EOF {
			break
		} else if err != nil {
			continue
		}
		p := classifier.GetPrediction(&instance)

		if r, ok := eval_result[p]; ok {
			r = new(Coordinate)
			(*r).n_pos = instance.pos_y
			(*r).n_neg = instance.neg_y
			(*r).p = p
			eval_result[p] = r
		} else {
			(*r).n_pos += instance.pos_y
			(*r).n_neg += instance.neg_y
		}
	}
	result := Coordinates(make([]*Coordinate, len(eval_result)))
	i := 0
	for _, v := range eval_result {
		result[i] = v
		i++
	}
	sort.Sort(result)
	return result
}

// RMSE returns the root mean squared error.
func RMSE(classifier BinaryClassifier, data_accessor *DataInstanceAccessor) float64 {

	return 0
}

// LogLikelihood returns the log likelihood of the classifier fitting the given data.
func LogLikelihood(classifier BinaryClassifier, data_accessor DataInstanceAccessor) float64 {
	loglikelihood := float64(0)
	ForEachValidDataInstance(data_accessor, func(instance DataInstance) {
		p := float64(classifier.GetPrediction(&instance))
		if instance.pos_y > 0 {
			loglikelihood += float64(instance.pos_y) * math.Log(p)
		}
		if instance.neg_y > 0 {
			loglikelihood += float64(instance.neg_y) * math.Log(1-p)
		}
	})
	return loglikelihood
}

// L2NormOfParamaeters returns the L2 norm of all the parameters of the given RBM.
func (rbm *SparseClassRBM) L2NormOfParamaeters() float64 {
	return 0
}

func Sparsity(w interface{}, z interface{}) (zeros int, count int) {
	zeros, count = 0, 0
	z_t := reflect.TypeOf(z)
	w_t := reflect.TypeOf(w)
	w_v := reflect.ValueOf(w)

	switch w_t.Kind() {
	case reflect.Array, reflect.Slice:
		for i := 0; i < w_v.Len(); i++ {
			cur_z, cur_c := Sparsity(w_v.Index(i).Interface(), z)
			zeros += cur_z
			count += cur_c
		}
		break
	case z_t.Kind():
		if w == z {
			zeros, count = 1, 1
		} else {
			zeros, count = 0, 1
		}
		break
	default:
		panic(fmt.Sprintf("Invalid parameter w: %v.", w))
		break
	}
	return
}

func (rbm *SparseClassRBM) SparsityOfW() float64 {
	zero_weights, total_weights := Sparsity(rbm.w, WeightT(0))
	return float64(zero_weights) / float64(total_weights)
}

func (rbm *SparseClassRBM) SparsityOfU() float64 {
	zero_weights, total_weights := Sparsity(rbm.u, WeightT(0))
	return float64(zero_weights) / float64(total_weights)
}

func ForEachValidDataInstance(data_accessor DataInstanceAccessor, f func(DataInstance)) {
	data_accessor.Reset()
	for {
		instance, err := data_accessor.NextInstance()
		if err == io.EOF {
			break
		} else if err != nil {
			continue
		} else {
			f(instance)
		}
	}
}
