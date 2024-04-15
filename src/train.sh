#!/usr/bin/bash
python /workdir/src/train.py \
-e 10 \
--lr 1e-4 \
-d /workdir/mount/data \
-s /workdir/mount/result/test \
