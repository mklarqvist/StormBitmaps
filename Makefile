###################################################################
# Copyright (c) 2019
# Author(s): Marcus D. R. Klarqvist
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
###################################################################

OPTFLAGS  := -O3 -march=native
CFLAGS     = -std=c99 $(OPTFLAGS) $(DEBUG_FLAGS)
CPPFLAGS   = -std=c++0x $(OPTFLAGS) $(DEBUG_FLAGS)
CPP_SOURCE = benchmark.cpp classes.cpp
C_SOURCE   = storm.c experimental.c
OBJECTS    = $(CPP_SOURCE:.cpp=.o) $(C_SOURCE:.c=.o)

# Default target
all: benchmark

# Generic rules
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $<

benchmark.o: benchmark.cpp classes.h libalgebra.h classes.o storm.h storm.o experimental.o experimental.h
	$(CXX) $(CPPFLAGS) -I/home/marcus/CRoaring/include -c -o $@ $<

benchmark: benchmark.o classes.h libalgebra.h classes.o storm.h storm.o experimental.h experimental.o
	$(CXX) $(CPPFLAGS) -L/home/marcus/CRoaring/ storm.o experimental.o classes.o benchmark.o -o benchmark -lroaring

clean:
	rm -f $(OBJECTS)
	rm -f benchmark

.PHONY: all clean
