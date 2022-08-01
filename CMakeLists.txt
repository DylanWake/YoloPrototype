cmake_minimum_required(VERSION 3.22)
project(YoloPrototype CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(TORCH_DIR "D:\\Dependencies\\libtorch")

find_package(Torch REQUIRED PATHS ${TORCH_DIR} NO_DEFAULT_PATH)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${Torch_INCLUDE_DIRS})

add_executable(YoloPrototype main.cu base/ResnetBTNBlock.cu base/ResnetBTNBlock.cuh)

target_link_libraries(YoloPrototype ${TORCH_LIBRARIES})

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET YoloPrototype
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:YoloPrototype>)
endif (MSVC)

set_target_properties(YoloPrototype PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 86)
