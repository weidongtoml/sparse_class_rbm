// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package platform

import (
	"testing"
)

func Test_Master(t *testing.T) {
	master := NewModelServer("")
	master.Start()
}
