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
include useGeometry/CMakeFiles/eigenGeometry.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include useGeometry/CMakeFiles/eigenGeometry.dir/compiler_depend.make

# Include the progress variables for this target.
include useGeometry/CMakeFiles/eigenGeometry.dir/progress.make

# Include the compile flags for this target's objects.
include useGeometry/CMakeFiles/eigenGeometry.dir/flags.make

useGeometry/CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.o: useGeometry/CMakeFiles/eigenGeometry.dir/flags.make
useGeometry/CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.o: /home/jiaxi/slambook/ch3/useGeometry/eigenGeometry.cpp
useGeometry/CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.o: useGeometry/CMakeFiles/eigenGeometry.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jiaxi/slambook/ch3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object useGeometry/CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.o"
	cd /home/jiaxi/slambook/ch3/build/useGeometry && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT useGeometry/CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.o -MF CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.o.d -o CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.o -c /home/jiaxi/slambook/ch3/useGeometry/eigenGeometry.cpp

useGeometry/CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.i"
	cd /home/jiaxi/slambook/ch3/build/useGeometry && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jiaxi/slambook/ch3/useGeometry/eigenGeometry.cpp > CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.i

useGeometry/CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.s"
	cd /home/jiaxi/slambook/ch3/build/useGeometry && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jiaxi/slambook/ch3/useGeometry/eigenGeometry.cpp -o CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.s

# Object files for target eigenGeometry
eigenGeometry_OBJECTS = \
"CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.o"

# External object files for target eigenGeometry
eigenGeometry_EXTERNAL_OBJECTS =

bin/eigenGeometry: useGeometry/CMakeFiles/eigenGeometry.dir/eigenGeometry.cpp.o
bin/eigenGeometry: useGeometry/CMakeFiles/eigenGeometry.dir/build.make
bin/eigenGeometry: useGeometry/CMakeFiles/eigenGeometry.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/jiaxi/slambook/ch3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/eigenGeometry"
	cd /home/jiaxi/slambook/ch3/build/useGeometry && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/eigenGeometry.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
useGeometry/CMakeFiles/eigenGeometry.dir/build: bin/eigenGeometry
.PHONY : useGeometry/CMakeFiles/eigenGeometry.dir/build

useGeometry/CMakeFiles/eigenGeometry.dir/clean:
	cd /home/jiaxi/slambook/ch3/build/useGeometry && $(CMAKE_COMMAND) -P CMakeFiles/eigenGeometry.dir/cmake_clean.cmake
.PHONY : useGeometry/CMakeFiles/eigenGeometry.dir/clean

useGeometry/CMakeFiles/eigenGeometry.dir/depend:
	cd /home/jiaxi/slambook/ch3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiaxi/slambook/ch3 /home/jiaxi/slambook/ch3/useGeometry /home/jiaxi/slambook/ch3/build /home/jiaxi/slambook/ch3/build/useGeometry /home/jiaxi/slambook/ch3/build/useGeometry/CMakeFiles/eigenGeometry.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : useGeometry/CMakeFiles/eigenGeometry.dir/depend
