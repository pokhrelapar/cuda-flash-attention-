set(SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Collect source files
file(GLOB CUDA_SOURCES "${SRC_PATH}/*.cu")
file(GLOB DATAGEN_SOURCES "${SRC_PATH}/dataset_generator.cpp" "${SRC_PATH}/fa_cpu.cpp")

# Create main executable
add_executable(fa_test ${CUDA_SOURCES} "${SRC_PATH}/main_test.cpp")
target_include_directories(fa_test PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_directories(fa_test PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(fa_test ${COMMON_LIBS})
add_dependencies(fa_test libgputk)

# Create datagen executable
add_executable(fa_datagen ${DATAGEN_SOURCES})
target_include_directories(fa_datagen PRIVATE ${COMMON_INCLUDE_DIRS})
target_include_directories(fa_datagen PRIVATE /usr/local/libgputk/include)
target_link_directories(fa_datagen PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(fa_datagen ${COMMON_LIBS})
add_dependencies(fa_datagen libgputk)

# Create benchmark executable
add_executable(fa_benchmark ${CUDA_SOURCES} "${SRC_PATH}/main_benchmark.cpp")
target_include_directories(fa_benchmark PRIVATE ${COMMON_INCLUDE_DIRS})
target_include_directories(fa_benchmark PRIVATE /usr/local/libgputk/include)
target_link_directories(fa_benchmark PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(fa_benchmark ${COMMON_LIBS})
add_dependencies(fa_benchmark libgputk)

# Set output directories
set_target_properties(fa_test fa_datagen fa_benchmark
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/release")