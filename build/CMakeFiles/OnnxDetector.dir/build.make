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
CMAKE_COMMAND = /home/wxxz/anaconda3/envs/yolov5/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/wxxz/anaconda3/envs/yolov5/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wxxz/workspace/OnnxruntimeDetector

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wxxz/workspace/OnnxruntimeDetector/build

# Include any dependencies generated for this target.
include CMakeFiles/OnnxDetector.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/OnnxDetector.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/OnnxDetector.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/OnnxDetector.dir/flags.make

CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.o: CMakeFiles/OnnxDetector.dir/flags.make
CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.o: CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp
CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.o: CMakeFiles/OnnxDetector.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/wxxz/workspace/OnnxruntimeDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.o -MF CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.o.d -o CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.o -c /home/wxxz/workspace/OnnxruntimeDetector/build/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp

CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wxxz/workspace/OnnxruntimeDetector/build/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp > CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.i

CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wxxz/workspace/OnnxruntimeDetector/build/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp -o CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.s

CMakeFiles/OnnxDetector.dir/main.cpp.o: CMakeFiles/OnnxDetector.dir/flags.make
CMakeFiles/OnnxDetector.dir/main.cpp.o: /home/wxxz/workspace/OnnxruntimeDetector/main.cpp
CMakeFiles/OnnxDetector.dir/main.cpp.o: CMakeFiles/OnnxDetector.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/wxxz/workspace/OnnxruntimeDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/OnnxDetector.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OnnxDetector.dir/main.cpp.o -MF CMakeFiles/OnnxDetector.dir/main.cpp.o.d -o CMakeFiles/OnnxDetector.dir/main.cpp.o -c /home/wxxz/workspace/OnnxruntimeDetector/main.cpp

CMakeFiles/OnnxDetector.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/OnnxDetector.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wxxz/workspace/OnnxruntimeDetector/main.cpp > CMakeFiles/OnnxDetector.dir/main.cpp.i

CMakeFiles/OnnxDetector.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/OnnxDetector.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wxxz/workspace/OnnxruntimeDetector/main.cpp -o CMakeFiles/OnnxDetector.dir/main.cpp.s

CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.o: CMakeFiles/OnnxDetector.dir/flags.make
CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.o: /home/wxxz/workspace/OnnxruntimeDetector/yolov5/ModelParser.cpp
CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.o: CMakeFiles/OnnxDetector.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/wxxz/workspace/OnnxruntimeDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.o -MF CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.o.d -o CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.o -c /home/wxxz/workspace/OnnxruntimeDetector/yolov5/ModelParser.cpp

CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wxxz/workspace/OnnxruntimeDetector/yolov5/ModelParser.cpp > CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.i

CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wxxz/workspace/OnnxruntimeDetector/yolov5/ModelParser.cpp -o CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.s

CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.o: CMakeFiles/OnnxDetector.dir/flags.make
CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.o: /home/wxxz/workspace/OnnxruntimeDetector/yolov5/ModelProcessor.cpp
CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.o: CMakeFiles/OnnxDetector.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/wxxz/workspace/OnnxruntimeDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.o -MF CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.o.d -o CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.o -c /home/wxxz/workspace/OnnxruntimeDetector/yolov5/ModelProcessor.cpp

CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wxxz/workspace/OnnxruntimeDetector/yolov5/ModelProcessor.cpp > CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.i

CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wxxz/workspace/OnnxruntimeDetector/yolov5/ModelProcessor.cpp -o CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.s

CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.o: CMakeFiles/OnnxDetector.dir/flags.make
CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.o: /home/wxxz/workspace/OnnxruntimeDetector/yolov5/Yolov5Session.cpp
CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.o: CMakeFiles/OnnxDetector.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/wxxz/workspace/OnnxruntimeDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.o -MF CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.o.d -o CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.o -c /home/wxxz/workspace/OnnxruntimeDetector/yolov5/Yolov5Session.cpp

CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wxxz/workspace/OnnxruntimeDetector/yolov5/Yolov5Session.cpp > CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.i

CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wxxz/workspace/OnnxruntimeDetector/yolov5/Yolov5Session.cpp -o CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.s

# Object files for target OnnxDetector
OnnxDetector_OBJECTS = \
"CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.o" \
"CMakeFiles/OnnxDetector.dir/main.cpp.o" \
"CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.o" \
"CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.o" \
"CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.o"

# External object files for target OnnxDetector
OnnxDetector_EXTERNAL_OBJECTS =

OnnxDetector: CMakeFiles/OnnxDetector.dir/CMakeFiles/3.27.4/CompilerIdCXX/CMakeCXXCompilerId.cpp.o
OnnxDetector: CMakeFiles/OnnxDetector.dir/main.cpp.o
OnnxDetector: CMakeFiles/OnnxDetector.dir/yolov5/ModelParser.cpp.o
OnnxDetector: CMakeFiles/OnnxDetector.dir/yolov5/ModelProcessor.cpp.o
OnnxDetector: CMakeFiles/OnnxDetector.dir/yolov5/Yolov5Session.cpp.o
OnnxDetector: CMakeFiles/OnnxDetector.dir/build.make
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
OnnxDetector: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
OnnxDetector: CMakeFiles/OnnxDetector.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/wxxz/workspace/OnnxruntimeDetector/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable OnnxDetector"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/OnnxDetector.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/OnnxDetector.dir/build: OnnxDetector
.PHONY : CMakeFiles/OnnxDetector.dir/build

CMakeFiles/OnnxDetector.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/OnnxDetector.dir/cmake_clean.cmake
.PHONY : CMakeFiles/OnnxDetector.dir/clean

CMakeFiles/OnnxDetector.dir/depend:
	cd /home/wxxz/workspace/OnnxruntimeDetector/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wxxz/workspace/OnnxruntimeDetector /home/wxxz/workspace/OnnxruntimeDetector /home/wxxz/workspace/OnnxruntimeDetector/build /home/wxxz/workspace/OnnxruntimeDetector/build /home/wxxz/workspace/OnnxruntimeDetector/build/CMakeFiles/OnnxDetector.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/OnnxDetector.dir/depend

