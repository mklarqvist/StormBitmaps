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
CPP_SOURCE = fast_intersect_count.cpp main.cpp
C_SOURCE   = 
OBJECTS    = $(CPP_SOURCE:.cpp=.o) $(C_SOURCE:.c=.o)

# Default target
all: intersect

# Generic rules
%.o: %.c
	$(CC) $(CFLAGS)-c -o $@ $<

%.o: %.cpp
	$(CXX) $(CPPFLAGS)-c -o $@ $<

main.o: main.cpp classes.h fast_intersect_count.h
	$(CXX) $(CPPFLAGS)-c -o $@ $<

intersect: fast_intersect_count.o main.o classes.h fast_intersect_count.h
	$(CXX) $(CPPFLAGS) -L/home/mdrk/tools/CRoaring/build fast_intersect_count.o main.o -o intersect -lroaring

clean:
	rm -f $(OBJECTS)
	rm -f intersect

.PHONY: all clean