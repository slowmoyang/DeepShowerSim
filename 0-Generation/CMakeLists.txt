# $Id: CMakeLists.txt 86065 2014-11-07 08:51:15Z gcosmo $

#
# usage example:
# cmake -DGeant4_DIR=$G4LIB/Geant4.<XX.YY> -DCMAKE_CXX_COMPILER=g++ <path-to-your-source-code>
# make
#

#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(generate)

#----------------------------------------------------------------------------
# Find Geant4 package, activating all available UI and Vis drivers by default
# You can set WITH_GEANT4_UIVIS to OFF via the command line or ccmake/cmake-gui
# to build a batch mode only executable
#
option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all multithreaded)
else()
  find_package(Geant4 REQUIRED)
endif()

# find_package(Boost COMPONENTS program_options)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED)


#----------------------------------------------------------------------------
# Setup Geant4 include directories and compile definitions
# Setup include directory for this project
#
include(${Geant4_USE_FILE})
include_directories(${PROJECT_SOURCE_DIR}/include)


#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

#----------------------------------------------------------------------------
# Add the executable, and link it to the Geant4 libraries
#
add_executable(generate generate.cc ${sources} ${headers})
target_link_libraries(generate ${Geant4_LIBRARIES})
# target_link_libraries(generate ${Boost_LIBRARIES})
target_link_libraries(generate Boost::program_options)

#----------------------------------------------------------------------------
# Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
#
install(TARGETS generate DESTINATION bin)
