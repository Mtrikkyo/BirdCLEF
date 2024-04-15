#!/usr/bin/bash
python /workdir/src/train.py \
-e 10 \
--lr 1e-4 \
-b 32 \
-d /workdir/mount/data \
-s /workdir/mount/result/test \
