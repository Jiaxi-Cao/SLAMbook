# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jiaxi/slambook/ch3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jiaxi/slambook/ch3/build

# Include any dependencies generated for this target.
include useEigen/CMakeFiles/eigenMatrix.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include useEigen/CMakeFiles/eigenMatrix.dir/compiler_depend.make

# Include the progress variables for this target.
include useEigen/CMakeFiles/eigenMatrix.dir/progress.make

# Include the compile flags for this target's objects.
include useEigen/CMakeFiles/eigenMatrix.dir/flags.make

useEigen/CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.o: useEigen/CMakeFiles/eigenMatrix.dir/flags.make
useEigen/CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.o: /home/jiaxi/slambook/ch3/useEigen/eigenMatrix.cpp
useEigen/CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.o: useEigen/CMakeFiles/eigenMatrix.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jiaxi/slambook/ch3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object useEigen/CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.o"
	cd /home/jiaxi/slambook/ch3/build/useEigen && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT useEigen/CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.o -MF CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.o.d -o CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.o -c /home/jiaxi/slambook/ch3/useEigen/eigenMatrix.cpp

useEigen/CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.i"
	cd /home/jiaxi/slambook/ch3/build/useEigen && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaxi/slambook/ch3/useEigen/eigenMatrix.cpp > CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.i

useEigen/CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.s"
	cd /home/jiaxi/slambook/ch3/build/useEigen && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaxi/slambook/ch3/useEigen/eigenMatrix.cpp -o CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.s

# Object files for target eigenMatrix
eigenMatrix_OBJECTS = \
"CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.o"

# External object files for target eigenMatrix
eigenMatrix_EXTERNAL_OBJECTS =

bin/eigenMatrix: useEigen/CMakeFiles/eigenMatrix.dir/eigenMatrix.cpp.o
bin/eigenMatrix: useEigen/CMakeFiles/eigenMatrix.dir/build.make
bin/eigenMatrix: useEigen/CMakeFiles/eigenMatrix.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/jiaxi/slambook/ch3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/eigenMatrix"
	cd /home/jiaxi/slambook/ch3/build/useEigen && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/eigenMatrix.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
useEigen/CMakeFiles/eigenMatrix.dir/build: bin/eigenMatrix
.PHONY : useEigen/CMakeFiles/eigenMatrix.dir/build

useEigen/CMakeFiles/eigenMatrix.dir/clean:
	cd /home/jiaxi/slambook/ch3/build/useEigen && $(CMAKE_COMMAND) -P CMakeFiles/eigenMatrix.dir/cmake_clean.cmake
.PHONY : useEigen/CMakeFiles/eigenMatrix.dir/clean

useEigen/CMakeFiles/eigenMatrix.dir/depend:
	cd /home/jiaxi/slambook/ch3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiaxi/slambook/ch3 /home/jiaxi/slambook/ch3/useEigen /home/jiaxi/slambook/ch3/build /home/jiaxi/slambook/ch3/build/useEigen /home/jiaxi/slambook/ch3/build/useEigen/CMakeFiles/eigenMatrix.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : useEigen/CMakeFiles/eigenMatrix.dir/depend
