#!/bin/bash
sudo systemctl restart nvargus-daemon
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
