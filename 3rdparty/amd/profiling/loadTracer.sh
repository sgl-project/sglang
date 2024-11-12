#!/bin/bash
################################################################################
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################
OUTPUT_FILE="trace.rpd"

if [ "$1" = "-o" ] ; then
  OUTPUT_FILE=$2
  shift
  shift
fi

if [ -e ${OUTPUT_FILE} ] ; then
  rm ${OUTPUT_FILE}
fi

python3 -m rocpd.schema --create ${OUTPUT_FILE}
if [ $? != 0 ] ; then
  echo "Error: Could not create rpd file. Please run 'python setup.py install' from the rocpd_python dir"
  exit
fi

export RPDT_FILENAME=${OUTPUT_FILE}
export RPDT_AUTOSTART=0
LD_PRELOAD=librocm-smi_64:librpd_tracer.so "$@"
