#!/bin/bash

apt-get update
apt-get install rsync
apt-get install ssh

mkdir -p ~/.ssh
touch ~/.ssh/config
chmod 600 ~/.ssh/config
echo -e "Host sdge\n\tHostname sdge.sdsc.edu\n\tUser tcaglar\n\tPort 21\n\tServerAliveInterval 60" > ~/.ssh/config

echo "Copying data_lite"
rsync -avz /home/pgmlvol/data/data_lite sdge:/qumulo/pgml-qf/ > ./rsync_data_lite.log 2>&1

echo "Copying ignition_patterns"
rsync -avz /home/pgmlvol/data/ignition_patterns sdge:/qumulo/pgml-qf/ > ./rsync_ignition_patterns.log 2>&1

echo "Copying data_lite_new"
rsync -avz /home/pgmlvol/data/data_lite_new sdge:/qumulo/pgml-qf/ > ./rsync_data_lite_new.log 2>&1

echo "Copying data_full"
rsync -avz /home/pgmlvol/data/data_full sdge:/qumulo/pgml-qf/ > ./rsync_data_full.log 2>&1