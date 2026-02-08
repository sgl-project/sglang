# A CMake script to find SetEnv.cmd.

find_program(WINSDK_SETENV NAMES SetEnv.cmd
  PATHS "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Microsoft SDKs\\Windows;CurrentInstallFolder]/bin")
if (WINSDK_SETENV AND PRINT_PATH)
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo "${WINSDK_SETENV}")
endif ()
