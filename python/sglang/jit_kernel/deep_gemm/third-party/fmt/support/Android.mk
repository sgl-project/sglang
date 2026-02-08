LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

LOCAL_MODULE := fmt_static
LOCAL_MODULE_FILENAME := libfmt

LOCAL_SRC_FILES := ../src/format.cc

LOCAL_C_INCLUDES := $(LOCAL_PATH)
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)

LOCAL_CFLAGS += -std=c++11 -fexceptions

include $(BUILD_STATIC_LIBRARY)

