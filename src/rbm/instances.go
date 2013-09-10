// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
)

// DataInstance is used for storing data sample for training and prediction.
// In the case of prediction, the value of y is ignored.
type DataInstance struct {
	x     []int //values of each classes, in the order of Class0, Class1, ...
	pos_y int   //number of positive instances
	neg_y int   //number of negative instances
}

// Equal determines whether the given two DataInstance are equal.
func (instance *DataInstance) Equal(a *DataInstance) bool {
	if !(instance.pos_y == a.pos_y && instance.neg_y == a.neg_y &&
		len(instance.x) == len(a.x)) {
		return false
	}
	for i, v := range instance.x {
		if v != a.x[i] {
			return false
		}
	}
	return true
}

// DataInstanceAccessor specifies an interface for retrieving training
// and testing data.
type DataInstanceAccessor interface {
	Reset()
	NextInstance() (DataInstance, error)
	Close()
}

// SequentialDataLoader implements the DataInstanceAccessor interface, and supplies
// training instances in a sequential manner.
type SequentialDataLoader struct {
	filename    string
	file        *os.File
	reader      *bufio.Reader
	num_classes int
}

// NewInstanceLoader creates an InstnaceLoader from the given file, with each
// instance having num_feature_class number of features.
func NewInstanceLoader(filename string, num_feature_class int) *SequentialDataLoader {
	file, err := os.Open(filename)
	if err != nil {
		log.Printf("Failed to open file: %s. %s.", filename, err)
		return nil
	}
	return &SequentialDataLoader{filename, file, bufio.NewReader(file), num_feature_class}
}

// Reset resets the underlying file cursor.
func (loader *SequentialDataLoader) Reset() {
	if _, err := loader.file.Seek(0, 0); err != nil {
		log.Printf("Failed to reset file %s: %s.", loader.filename, err)
	}
}

// Close closes the underlying file.
func (loader *SequentialDataLoader) Close() {
	loader.reader = nil
	loader.file.Close()
}

// NextInstance retrieves the next data instance, return EOF when end of file
// has been reached, error if other errors have been encountered, or nil and
// a valid DataInstance when everything is fine.
func (loader *SequentialDataLoader) NextInstance() (DataInstance, error) {
	var instance DataInstance
	line, err := loader.reader.ReadString('\n')
	if err != nil {
		if err == io.EOF {
			return instance, io.EOF
		} else {
			return instance, fmt.Errorf("Failed to retrieve instance: %s.", err)
		}
	}
	line = strings.Trim(line, "\n\t\r\f")
	fields := strings.Split(line, "\t")
	if len(fields) < 3 {
		return instance, fmt.Errorf("Expected each instance to have at least 3 fields: %s.", line)
	}
	pos_y_cnt, err := strconv.ParseUint(fields[0], 10, 16)
	if err != nil {
		return instance, fmt.Errorf("Expected postitive instance counnt: %s.", line)
	}
	neg_y_cnt, err := strconv.ParseUint(fields[1], 10, 16)
	if err != nil {
		return instance, fmt.Errorf("Expected negative instance count: %s.", line)
	}
	feature_sets := make([]int, loader.num_classes)
	for _, v := range fields[2:] {
		feature := strings.Split(v, ":")
		if len(feature) != 2 {
			return instance, fmt.Errorf("Invalid feature: %s.", feature)
		}
		class_id, err := strconv.ParseInt(feature[0], 10, 16)
		if err != nil {
			return instance, fmt.Errorf("Expected class_id to be integer but got %s.", feature[0])
		}
		class_val, err := strconv.ParseInt(feature[1], 10, 16)
		if err != nil {
			return instance, fmt.Errorf("Expected class_val to be integer but got %s.", feature[1])
		}
		if int(class_id) > len(feature_sets) {
			return instance, fmt.Errorf("Error, expected max class id to be %d but got %d.",
				len(feature_sets)-1, class_id)
		}
		feature_sets[int(class_id)] = int(class_val)
	}
	instance.pos_y = int(pos_y_cnt)
	instance.neg_y = int(neg_y_cnt)
	instance.x = feature_sets
	return instance, nil
}

func GetBiases(class_sizes []int, accessor DataInstanceAccessor) ([][]WeightT, WeightT) {
	biases := make([][]WeightT, len(class_sizes))
	for i, s := range class_sizes {
		biases[i] = make([]WeightT, s)
	}

	pos_y := 0
	neg_y := 0
	for {
		inst, err := accessor.NextInstance()
		if err == io.EOF {
			break
		} else if err == nil {
			for i, v := range inst.x {
				biases[i][v]++
			}
			pos_y += inst.pos_y
			neg_y += inst.neg_y
		}
	}

	for i := range biases {
		s := WeightT(0)
		for _, v := range biases[i] {
			s += v
		}
		for j, _ := range biases[i] {
			biases[i][j] /= s
		}
	}

	return biases, WeightT(pos_y) / WeightT(pos_y+neg_y)
}
