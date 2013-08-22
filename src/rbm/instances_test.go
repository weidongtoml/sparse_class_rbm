// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rbm

import (
	"bufio"
	"common/util"
	"fmt"
	"io"
	"os"
	"testing"
)

func Test_SequentialDataLoader(t *testing.T) {
	test_file := "test_instances.dat"
	num_feature_classes := 6
	test_cases := []DataInstance{
		{
			[]int{0, 1, 2, 3, 4, 5},
			1,
			2,
		}, {
			[]int{1, 2, 3, 4, 5, 0},
			0,
			1,
		},
	}
	err := util.WithNewOpenFileAsBufioWriter(test_file,
		func(w *bufio.Writer) error {
			for _, t_case := range test_cases {
				fmt.Fprintf(w, "%d\t%d", t_case.pos_y, t_case.neg_y)
				for i, v := range t_case.x {
					fmt.Fprintf(w, "\t%d:%d", i, v)
				}
				fmt.Fprint(w, "\n")
			}
			return nil
		})
	defer func() {
		os.Remove(test_file)
	}()
	if err != nil {
		t.Errorf("Failed to create test file: %s:%s.", test_file, err)
	}
	loader := NewInstanceLoader(test_file, num_feature_classes)
	for i, t_case := range test_cases {
		instance, err := loader.NextInstance()
		if err != nil {
			t.Errorf("TestCase: #%d: %s.", i, err)
		}
		if !(&instance).Equal(&t_case) {
			t.Errorf("TestCase: #%d: Expected %v but got %v.", i, t_case, instance)
		}
	}
	_, err = loader.NextInstance()
	if err != io.EOF {
		t.Errorf("Expected EOF but got %s.", err)
	}
	loader.Reset()
	_, err = loader.NextInstance()
	if err != nil {
		t.Errorf("Expected instance but got error.")
	}
	loader.Close()
}
