set(CMAKE_CXX_STANDARD 14)

add_executable(run_io_metrics run_io_metrics.cpp)
target_link_libraries(run_io_metrics ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)
