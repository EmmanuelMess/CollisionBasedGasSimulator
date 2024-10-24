#! /bin/bash

rm -f simulator.c

touch simulator.c

cat opencl_header.c.in > simulator.c

cat "simulator.cl" | sed -z 's/\\n/\\\\n/g' | sed -z 's/"/\\"/g' | sed -z 's/\n/\\n"\n"/g' >> simulator.c

cat opencl_footer.c.in >> simulator.c