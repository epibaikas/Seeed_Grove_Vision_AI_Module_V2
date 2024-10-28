#!/bin/sh

echo "Flash device using python script"

SERIAL_PORT="/dev/cu.usbmodem578D0263771"
python ../xmodem/xmodem_send.py --port=$SERIAL_PORT --baudrate=921600 --protocol=xmodem --file=../we2_image_gen_local/output_case1_sec_wlcsp/output.img
