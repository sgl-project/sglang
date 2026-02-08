// Formatting library for C++ - mocks of POSIX functions
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#ifndef FMT_POSIX_TEST_H
#define FMT_POSIX_TEST_H

#include <errno.h>
#include <locale.h>
#include <stdio.h>
#ifdef __APPLE__
#  include <xlocale.h>
#endif

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/param.h>  // for FreeBSD version
#  include <sys/types.h>  // for ssize_t
#endif

#ifndef _MSC_VER
struct stat;
#endif

namespace test {

#ifndef _MSC_VER
// Size type for read and write.
using size_t = size_t;
using ssize_t = ssize_t;
int open(const char* path, int oflag, int mode);
int fstat(int fd, struct stat* buf);
#else
using size_t = unsigned;
using ssize_t = int;
#endif

#ifndef _WIN32
long sysconf(int name);
#else
DWORD GetFileSize(HANDLE hFile, LPDWORD lpFileSizeHigh);
#endif

int close(int fildes);

int dup(int fildes);
int dup2(int fildes, int fildes2);

FILE* fdopen(int fildes, const char* mode);

ssize_t read(int fildes, void* buf, size_t nbyte);
ssize_t write(int fildes, const void* buf, size_t nbyte);

#ifndef _WIN32
int pipe(int fildes[2]);
#else
int pipe(int* pfds, unsigned psize, int textmode);
#endif

FILE* fopen(const char* filename, const char* mode);
int fclose(FILE* stream);
int(fileno)(FILE* stream);

#if defined(FMT_LOCALE) && !defined(_WIN32)
locale_t newlocale(int category_mask, const char* locale, locale_t base);
#endif
}  // namespace test

#define FMT_SYSTEM(call) test::call

#endif  // FMT_POSIX_TEST_H
