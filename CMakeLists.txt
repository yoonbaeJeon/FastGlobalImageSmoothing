cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_SUPPRESS_REGENERATION true)

# Build type
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Debug' as none was specified.")
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif ()

project(FGS)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
file(GLOB SOURCE_DIR
    "src/*.h"
    "src/*.hpp"
    "src/*.cpp"
)
add_executable(FGS ${SOURCE_DIR})
target_link_libraries(FGS ${OpenCV_LIBS})