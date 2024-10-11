#! /bin/bash

rm -f simulator.cl

touch simulator.cl

cat opencl_header.c.in > simulator.cl

cat "simulator.cl.in" | sed -z 's/\\n/\\\\n/g' | sed -z 's/"/\\"/g' | sed -z 's/\n/\\n"\n"/g' >> simulator.cl

cat opencl_footer.c.in >> simulator.cl