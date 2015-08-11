CC=g++
CCFLAGS=-Wall -fPIC -std=c++11 -c 
INCLUDES=-I./inc
DEFINITIONS=-DBOOST_LOG_DYN_LINK
LDFLAGS=-shared -lcblas -ljson-c -lboost_log
SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(addprefix obj/,$(notdir $(SOURCES:.cpp=.o)))
VPATH=src
LIB=lib/libnnet.so

all: $(LIB)

$(LIB): $(OBJECTS) 
	$(CC) -o $@ $^ $(LDFLAGS)

$(OBJECTS): obj/%.o: src/%.cpp 
	$(CC) $(INCLUDES) $(DEFINITIONS) $(CCFLAGS) $< -o $@

clean:
	$(RM) $(OBJECTS) $(LIB)

