// Copyright 2013 Weidong Liang. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package platform

import (
	"io"
	"net/http"
)

type ModelServer struct {
	local_addr string
}

func NewModelServer(config_file string) *ModelServer {
	var server ModelServer
	server.local_addr = ":8080"
	(&server).loadConfig(config_file)
	return &server
}

func (server *ModelServer) Start() {
	http.HandleFunc("/", server.serveHome)
	http.ListenAndServe(server.local_addr, nil)
}

func (server *ModelServer) loadConfig(config_file string) {

}

func (server *ModelServer) serveHome(w http.ResponseWriter, req *http.Request) {
	io.WriteString(w, "hello, world!\n")
}
