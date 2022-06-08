
# Android makefile
# Build this using ndk as
# ndk-build NDK_PROJECT_PATH=.  APP_BUILD_SCRIPT=Android.mk
#

LOCAL_PATH := $(call my-dir)
TVM_PATH := /home/wanger/code/tvm_ori

include $(CLEAR_VARS)
LOCAL_MODULE := libOpenCL

LOCAL_C_INCLUDES := $(LOCAL_PATH)/inc/ \
                    $(TVM_PATH)/include/ \
                    $(TVM_PATH)/3rdparty/dlpack/include/ \
                    $(TVM_PATH)/3rdparty/dmlc-core/include/ \

OPENCL_STUB_SRC_FILES := src/libopencl.cc

TVM_SRC_FILES := $(TVM_PATH)/src/runtime/c_runtime_api.cc                         \
                 $(TVM_PATH)/src/runtime/container.cc                             \
                 $(TVM_PATH)/src/runtime/cpu_device_api.cc                        \
                 $(TVM_PATH)/src/runtime/file_utils.cc                            \
                 $(TVM_PATH)/src/runtime/library_module.cc                        \
                 $(TVM_PATH)/src/runtime/logging.cc                               \
                 $(TVM_PATH)/src/runtime/module.cc                                \
                 $(TVM_PATH)/src/runtime/ndarray.cc                               \
                 $(TVM_PATH)/src/runtime/object.cc                                \
                 $(TVM_PATH)/src/runtime/registry.cc                              \
                 $(TVM_PATH)/src/runtime/thread_pool.cc                           \
                 $(TVM_PATH)/src/runtime/threading_backend.cc                     \
                 $(TVM_PATH)/src/runtime/workspace_pool.cc                        \
                 $(TVM_PATH)/src/runtime/dso_library.cc                           \
                 $(TVM_PATH)/src/runtime/system_library.cc                        \
                 $(TVM_PATH)/src/runtime/graph_executor/graph_executor.cc         \
                 $(TVM_PATH)/src/runtime/graph_executor/graph_executor_factory.cc \
                 $(TVM_PATH)/src/runtime/source_utils.cc                          \
                 $(TVM_PATH)/src/runtime/opencl/texture_pool.cc                   \
                 $(TVM_PATH)/src/runtime/opencl/opencl_device_api.cc              \
                 $(TVM_PATH)/src/runtime/opencl/opencl_module.cc                  \

LOCAL_SRC_FILES :=  src/cpp_deploy.cc         \
                    $(OPENCL_STUB_SRC_FILES)  \
                    $(TVM_SRC_FILES)          \

LOCAL_CFLAGS   = -fPIC -O2 -frtti -std=c++17
LOCAL_CPP_FEATURES += exceptions

include $(BUILD_EXECUTABLE)

