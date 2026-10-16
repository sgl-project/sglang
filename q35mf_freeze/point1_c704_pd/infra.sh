#!/bin/bash
rm -rf /tmp/nats /tmp/etcd; mkdir -p /tmp/nats
printf "max_payload: 8388608\njetstream { store_dir: \"/tmp/nats\" }\n" > /tmp/nats.conf; /configs/nats-server -c /tmp/nats.conf &
/configs/etcd --data-dir /tmp/etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://<P0_IP>:2379 &
wait -n; echo "infra process exited rc=$?"; kill $(jobs -p) 2>/dev/null; wait
