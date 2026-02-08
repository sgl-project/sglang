// Formatting library for C++ - time formatting tests
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include "fmt/chrono.h"

#include <algorithm>
#include <ctime>
#include <vector>

#include "gtest-extra.h"  // EXPECT_THROW_MSG
#include "util.h"         // get_locale

using fmt::runtime;
using fmt::sys_time;
using testing::Contains;

#if defined(__MINGW32__) && !defined(_UCRT)
// Only C89 conversion specifiers when using MSVCRT instead of UCRT
#  define FMT_HAS_C99_STRFTIME 0
#else
#  define FMT_HAS_C99_STRFTIME 1
#endif

#if defined(__cpp_lib_chrono) && __cpp_lib_chrono >= 201907L
using days = std::chrono::days;
#else
using days = std::chrono::duration<std::chrono::hours::rep, std::ratio<86400>>;
#endif

auto make_tm() -> std::tm {
  auto time = std::tm();
  time.tm_mday = 1;
  return time;
}

auto make_hour(int h) -> std::tm {
  auto time = make_tm();
  time.tm_hour = h;
  return time;
}

auto make_minute(int m) -> std::tm {
  auto time = make_tm();
  time.tm_min = m;
  return time;
}

auto make_second(int s) -> std::tm {
  auto time = make_tm();
  time.tm_sec = s;
  return time;
}

std::string system_strftime(const std::string& format, const std::tm* timeptr,
                            std::locale* locptr = nullptr) {
  auto loc = locptr ? *locptr : std::locale::classic();
  auto& facet = std::use_facet<std::time_put<char>>(loc);
  std::ostringstream os;
  os.imbue(loc);
  facet.put(os, os, ' ', timeptr, format.c_str(),
            format.c_str() + format.size());
#ifdef _WIN32
  // Workaround a bug in older versions of Universal CRT.
  auto str = os.str();
  if (str == "-0000") str = "+0000";
  return str;
#else
  return os.str();
#endif
}

FMT_CONSTEXPR std::tm make_tm(int year, int mon, int mday, int hour, int min,
                              int sec) {
  auto tm = std::tm();
  tm.tm_sec = sec;
  tm.tm_min = min;
  tm.tm_hour = hour;
  tm.tm_mday = mday;
  tm.tm_mon = mon - 1;
  tm.tm_year = year - 1900;
  return tm;
}

TEST(chrono_test, format_tm) {
  auto tm = std::tm();
  tm.tm_year = 116;
  tm.tm_mon = 3;
  tm.tm_mday = 25;
  tm.tm_hour = 11;
  tm.tm_min = 22;
  tm.tm_sec = 33;
  EXPECT_EQ(fmt::format("The date is {:%Y-%m-%d %H:%M:%S}.", tm),
            "The date is 2016-04-25 11:22:33.");
  EXPECT_EQ(fmt::format("{:%Y}", tm), "2016");
  EXPECT_EQ(fmt::format("{:%C}", tm), "20");
  EXPECT_EQ(fmt::format("{:%C%y}", tm), fmt::format("{:%Y}", tm));
  EXPECT_EQ(fmt::format("{:%e}", tm), "25");
  EXPECT_EQ(fmt::format("{:%D}", tm), "04/25/16");
  EXPECT_EQ(fmt::format("{:%F}", tm), "2016-04-25");
  EXPECT_EQ(fmt::format("{:%T}", tm), "11:22:33");

  // Short year
  tm.tm_year = 999 - 1900;
  tm.tm_mon = 0;   // for %G
  tm.tm_mday = 2;  // for %G
  tm.tm_wday = 3;  // for %G
  tm.tm_yday = 1;  // for %G
  EXPECT_EQ(fmt::format("{:%Y}", tm), "0999");
  EXPECT_EQ(fmt::format("{:%C%y}", tm), "0999");
  EXPECT_EQ(fmt::format("{:%G}", tm), "0999");

  tm.tm_year = 27 - 1900;
  EXPECT_EQ(fmt::format("{:%Y}", tm), "0027");
  EXPECT_EQ(fmt::format("{:%C%y}", tm), "0027");

  // Overflow year
  tm.tm_year = 2147483647;
  EXPECT_EQ(fmt::format("{:%Y}", tm), "2147485547");

  tm.tm_year = -2147483648;
  EXPECT_EQ(fmt::format("{:%Y}", tm), "-2147481748");

  // for week on the year
  // https://www.cl.cam.ac.uk/~mgk25/iso-time.html
  std::vector<std::tm> tm_list = {
      make_tm(1975, 12, 29, 12, 14, 16),  // W01
      make_tm(1977, 1, 2, 12, 14, 16),    // W53
      make_tm(1999, 12, 27, 12, 14, 16),  // W52
      make_tm(1999, 12, 31, 12, 14, 16),  // W52
      make_tm(2000, 1, 1, 12, 14, 16),    // W52
      make_tm(2000, 1, 2, 12, 14, 16),    // W52
      make_tm(2000, 1, 3, 12, 14, 16)     // W1
  };

#if !FMT_HAS_C99_STRFTIME
  GTEST_SKIP() << "Skip the rest of this test because it relies on strftime() "
                  "conforming to C99, but on this platform, MINGW + MSVCRT, "
                  "the function conforms only to C89.";
#endif

  const std::string iso_week_spec = "%Y-%m-%d: %G %g %V";
  for (auto ctm : tm_list) {
    // Calculate tm_yday, tm_wday, etc.
    std::time_t t = std::mktime(&ctm);
    tm = *std::localtime(&t);

    auto fmt_spec = fmt::format("{{:{}}}", iso_week_spec);
    EXPECT_EQ(system_strftime(iso_week_spec, &tm),
              fmt::format(fmt::runtime(fmt_spec), tm));
  }

  // Every day from 1970-01-01
  std::time_t time_now = std::time(nullptr);
  for (std::time_t t = 6 * 3600; t < time_now; t += 86400) {
    tm = *std::localtime(&t);

    auto fmt_spec = fmt::format("{{:{}}}", iso_week_spec);
    EXPECT_EQ(system_strftime(iso_week_spec, &tm),
              fmt::format(fmt::runtime(fmt_spec), tm));
  }
}

// MSVC:
//  minkernel\crts\ucrt\src\appcrt\time\wcsftime.cpp(971) : Assertion failed:
//  timeptr->tm_year >= -1900 && timeptr->tm_year <= 8099
#ifndef _WIN32
TEST(chrono_test, format_tm_future) {
  auto tm = std::tm();
  tm.tm_year = 10445;  // 10000+ years
  tm.tm_mon = 3;
  tm.tm_mday = 25;
  tm.tm_hour = 11;
  tm.tm_min = 22;
  tm.tm_sec = 33;
  EXPECT_EQ(fmt::format("The date is {:%Y-%m-%d %H:%M:%S}.", tm),
            "The date is 12345-04-25 11:22:33.");
  EXPECT_EQ(fmt::format("{:%Y}", tm), "12345");
  EXPECT_EQ(fmt::format("{:%C}", tm), "123");
  EXPECT_EQ(fmt::format("{:%C%y}", tm), fmt::format("{:%Y}", tm));
  EXPECT_EQ(fmt::format("{:%D}", tm), "04/25/45");
  EXPECT_EQ(fmt::format("{:%F}", tm), "12345-04-25");
  EXPECT_EQ(fmt::format("{:%T}", tm), "11:22:33");
}

TEST(chrono_test, format_tm_past) {
  auto tm = std::tm();
  tm.tm_year = -2001;
  tm.tm_mon = 3;
  tm.tm_mday = 25;
  tm.tm_hour = 11;
  tm.tm_min = 22;
  tm.tm_sec = 33;
  EXPECT_EQ(fmt::format("The date is {:%Y-%m-%d %H:%M:%S}.", tm),
            "The date is -101-04-25 11:22:33.");
  EXPECT_EQ(fmt::format("{:%Y}", tm), "-101");

  // macOS  %C - "-1"
  // Linux  %C - "-2"
  // fmt    %C - "-1"
  EXPECT_EQ(fmt::format("{:%C}", tm), "-1");
  EXPECT_EQ(fmt::format("{:%C%y}", tm), fmt::format("{:%Y}", tm));

  // macOS  %D - "04/25/01" (%y)
  // Linux  %D - "04/25/99" (%y)
  // fmt    %D - "04/25/01" (%y)
  EXPECT_EQ(fmt::format("{:%D}", tm), "04/25/01");

  EXPECT_EQ(fmt::format("{:%F}", tm), "-101-04-25");
  EXPECT_EQ(fmt::format("{:%T}", tm), "11:22:33");

  tm.tm_year = -1901;  // -1
  EXPECT_EQ(fmt::format("{:%Y}", tm), "-001");
  EXPECT_EQ(fmt::format("{:%C%y}", tm), fmt::format("{:%Y}", tm));

  tm.tm_year = -1911;  // -11
  EXPECT_EQ(fmt::format("{:%Y}", tm), "-011");
  EXPECT_EQ(fmt::format("{:%C%y}", tm), fmt::format("{:%Y}", tm));
}
#endif

TEST(chrono_test, grow_buffer) {
  auto s = std::string("{:");
  for (int i = 0; i < 30; ++i) s += "%c";
  s += "}\n";
  auto t = std::time(nullptr);
  (void)fmt::format(fmt::runtime(s), *std::localtime(&t));
}

TEST(chrono_test, format_to_empty_container) {
  auto time = std::tm();
  time.tm_sec = 42;
  auto s = std::string();
  fmt::format_to(std::back_inserter(s), "{:%S}", time);
  EXPECT_EQ(s, "42");
}

TEST(chrono_test, gmtime) {
  auto t = std::time(nullptr);
  auto expected = *std::gmtime(&t);
  auto actual = fmt::gmtime(t);
  EXPECT_EQ(actual.tm_sec, expected.tm_sec);
  EXPECT_EQ(actual.tm_min, expected.tm_min);
  EXPECT_EQ(actual.tm_hour, expected.tm_hour);
  EXPECT_EQ(actual.tm_mday, expected.tm_mday);
  EXPECT_EQ(actual.tm_mon, expected.tm_mon);
  EXPECT_EQ(actual.tm_year, expected.tm_year);
  EXPECT_EQ(actual.tm_wday, expected.tm_wday);
  EXPECT_EQ(actual.tm_yday, expected.tm_yday);
  EXPECT_EQ(actual.tm_isdst, expected.tm_isdst);
}

template <typename Time> void test_time(Time time) {
  EXPECT_EQ(fmt::format("{}", time), "1979-03-12 12:00:00");
  EXPECT_EQ(fmt::format("{:}", time), "1979-03-12 12:00:00");

  EXPECT_EQ(fmt::format("{:%%}", time), "%");
  EXPECT_EQ(fmt::format("{:%n}", time), "\n");
  EXPECT_EQ(fmt::format("{:%t}", time), "\t");
  EXPECT_EQ(fmt::format("{:%Y}", time), "1979");
  EXPECT_EQ(fmt::format("{:%EY}", time), "1979");
  EXPECT_EQ(fmt::format("{:%y}", time), "79");
  EXPECT_EQ(fmt::format("{:%Oy}", time), "79");
  EXPECT_EQ(fmt::format("{:%Ey}", time), "79");
  EXPECT_EQ(fmt::format("{:%C}", time), "19");
  EXPECT_EQ(fmt::format("{:%EC}", time), "19");
  EXPECT_EQ(fmt::format("{:%G}", time), "1979");
  EXPECT_EQ(fmt::format("{:%g}", time), "79");
  EXPECT_EQ(fmt::format("{:%b}", time), "Mar");
  EXPECT_EQ(fmt::format("{:%h}", time), "Mar");
  EXPECT_EQ(fmt::format("{:%B}", time), "March");
  EXPECT_EQ(fmt::format("{:%m}", time), "03");
  EXPECT_EQ(fmt::format("{:%Om}", time), "03");
  EXPECT_EQ(fmt::format("{:%U}", time), "10");
  EXPECT_EQ(fmt::format("{:%OU}", time), "10");
  EXPECT_EQ(fmt::format("{:%W}", time), "11");
  EXPECT_EQ(fmt::format("{:%OW}", time), "11");
  EXPECT_EQ(fmt::format("{:%V}", time), "11");
  EXPECT_EQ(fmt::format("{:%OV}", time), "11");
  EXPECT_EQ(fmt::format("{:%j}", time), "071");
  EXPECT_EQ(fmt::format("{:%d}", time), "12");
  EXPECT_EQ(fmt::format("{:%Od}", time), "12");
  EXPECT_EQ(fmt::format("{:%e}", time), "12");
  EXPECT_EQ(fmt::format("{:%Oe}", time), "12");
  EXPECT_EQ(fmt::format("{:%a}", time), "Mon");
  EXPECT_EQ(fmt::format("{:%A}", time), "Monday");
  EXPECT_EQ(fmt::format("{:%w}", time), "1");
  EXPECT_EQ(fmt::format("{:%Ow}", time), "1");
  EXPECT_EQ(fmt::format("{:%u}", time), "1");
  EXPECT_EQ(fmt::format("{:%Ou}", time), "1");
  EXPECT_EQ(fmt::format("{:%H}", time), "12");
  EXPECT_EQ(fmt::format("{:%OH}", time), "12");
  EXPECT_EQ(fmt::format("{:%I}", time), "12");
  EXPECT_EQ(fmt::format("{:%OI}", time), "12");
  EXPECT_EQ(fmt::format("{:%M}", time), "00");
  EXPECT_EQ(fmt::format("{:%OM}", time), "00");
  EXPECT_EQ(fmt::format("{:%S}", time), "00");
  EXPECT_EQ(fmt::format("{:%OS}", time), "00");
  EXPECT_EQ(fmt::format("{:%x}", time), "03/12/79");
  EXPECT_EQ(fmt::format("{:%Ex}", time), "03/12/79");
  EXPECT_EQ(fmt::format("{:%X}", time), "12:00:00");
  EXPECT_EQ(fmt::format("{:%EX}", time), "12:00:00");
  EXPECT_EQ(fmt::format("{:%D}", time), "03/12/79");
  EXPECT_EQ(fmt::format("{:%F}", time), "1979-03-12");
  EXPECT_EQ(fmt::format("{:%R}", time), "12:00");
  EXPECT_EQ(fmt::format("{:%T}", time), "12:00:00");
  EXPECT_EQ(fmt::format("{:%p}", time), "PM");
  EXPECT_EQ(fmt::format("{:%c}", time), "Mon Mar 12 12:00:00 1979");
  EXPECT_EQ(fmt::format("{:%Ec}", time), "Mon Mar 12 12:00:00 1979");
  EXPECT_EQ(fmt::format("{:%r}", time), "12:00:00 PM");

  EXPECT_EQ(fmt::format("{:%Y-%m-%d %H:%M:%S}", time), "1979-03-12 12:00:00");
}

TEST(chrono_test, sys_time) {
  auto time =
      fmt::sys_time<std::chrono::seconds>(std::chrono::seconds(290088000));
  test_time(time);
  EXPECT_EQ(fmt::format("{:%z}", time), "+0000");
  EXPECT_EQ(fmt::format("{:%Ez}", time), "+00:00");
  EXPECT_EQ(fmt::format("{:%Oz}", time), "+00:00");
  EXPECT_EQ(fmt::format("{:%Z}", time), "UTC");
}

TEST(chrono_test, local_time) {
  auto time =
      fmt::local_time<std::chrono::seconds>(std::chrono::seconds(290088000));
  test_time(time);
  EXPECT_THROW_MSG((void)fmt::format(fmt::runtime("{:%z}"), time),
                   fmt::format_error, "no timezone");
  EXPECT_THROW_MSG((void)fmt::format(fmt::runtime("{:%Z}"), time),
                   fmt::format_error, "no timezone");
}

template <typename T, FMT_ENABLE_IF(fmt::detail::has_tm_gmtoff<T>::value)>
bool set_tm_gmtoff(T& time, long offset) {
  time.tm_gmtoff = offset;
  return true;
}
template <typename T, FMT_ENABLE_IF(!fmt::detail::has_tm_gmtoff<T>::value)>
bool set_tm_gmtoff(T&, long) {
  return false;
}

TEST(chrono_test, tm) {
  auto time = fmt::gmtime(290088000);
  test_time(time);
  if (set_tm_gmtoff(time, -28800)) {
    EXPECT_EQ(fmt::format(fmt::runtime("{:%z}"), time), "-0800");
    EXPECT_EQ(fmt::format(fmt::runtime("{:%Ez}"), time), "-08:00");
    EXPECT_EQ(fmt::format(fmt::runtime("{:%Oz}"), time), "-08:00");
  } else {
    EXPECT_THROW_MSG((void)fmt::format(fmt::runtime("{:%z}"), time),
                     fmt::format_error, "no timezone");
  }
  char tz[] = "EET";
  if (fmt::detail::set_tm_zone(time, tz)) {
    EXPECT_EQ(fmt::format(fmt::runtime("{:%Z}"), time), "EET");
  } else {
    EXPECT_THROW_MSG((void)fmt::format(fmt::runtime("{:%Z}"), time),
                     fmt::format_error, "no timezone");
  }
}

TEST(chrono_test, daylight_savings_time_end) {
  // 2024-10-27 03:05 as the number of seconds since epoch in Europe/Kyiv time.
  // It is slightly after the DST end and passing it to to_sys will result in
  // an ambiguous time error:
  //   2024-10-27 03:05:00 is ambiguous.  It could be
  //   2024-10-27 03:05:00 EEST == 2024-10-27 00:05:00 UTC or
  //   2024-10-27 03:05:00 EET == 2024-10-27 01:05:00 UTC
  auto t =
      fmt::local_time<std::chrono::seconds>(std::chrono::seconds(1729998300));
  EXPECT_EQ(fmt::format("{}", t), "2024-10-27 03:05:00");
}

TEST(chrono_test, format_default) {
  EXPECT_EQ(fmt::format("{}", std::chrono::seconds(42)), "42s");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::atto>(42)),
            "42as");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::femto>(42)),
            "42fs");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::pico>(42)),
            "42ps");
  EXPECT_EQ(fmt::format("{}", std::chrono::nanoseconds(42)), "42ns");
  EXPECT_EQ(fmt::format("{}", std::chrono::microseconds(42)), "42µs");
  EXPECT_EQ(fmt::format("{}", std::chrono::milliseconds(42)), "42ms");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::centi>(42)),
            "42cs");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::deci>(42)),
            "42ds");
  EXPECT_EQ(fmt::format("{}", std::chrono::seconds(42)), "42s");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::deca>(42)),
            "42das");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::hecto>(42)),
            "42hs");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::kilo>(42)),
            "42ks");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::mega>(42)),
            "42Ms");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::giga>(42)),
            "42Gs");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::tera>(42)),
            "42Ts");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::peta>(42)),
            "42Ps");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<int, std::exa>(42)),
            "42Es");
  EXPECT_EQ(fmt::format("{}", std::chrono::minutes(42)), "42min");
  EXPECT_EQ(fmt::format("{}", std::chrono::hours(42)), "42h");
  EXPECT_EQ(fmt::format("{}", days(42)), "42d");
  EXPECT_EQ(
      fmt::format("{}", std::chrono::duration<int, std::ratio<15, 1>>(42)),
      "42[15]s");
  EXPECT_EQ(
      fmt::format("{}", std::chrono::duration<int, std::ratio<15, 4>>(42)),
      "42[15/4]s");
}

TEST(chrono_test, duration_align) {
  auto s = std::chrono::seconds(42);
  EXPECT_EQ(fmt::format("{:5}", s), "42s  ");
  EXPECT_EQ(fmt::format("{:{}}", s, 5), "42s  ");
  EXPECT_EQ(fmt::format("{:>5}", s), "  42s");
  EXPECT_EQ(fmt::format("{:*^7}", s), "**42s**");
  EXPECT_EQ(fmt::format("{:12%H:%M:%S}", std::chrono::seconds(12345)),
            "03:25:45    ");
  EXPECT_EQ(fmt::format("{:>12%H:%M:%S}", std::chrono::seconds(12345)),
            "    03:25:45");
  EXPECT_EQ(fmt::format("{:~^12%H:%M:%S}", std::chrono::seconds(12345)),
            "~~03:25:45~~");
  EXPECT_EQ(fmt::format("{:{}%H:%M:%S}", std::chrono::seconds(12345), 12),
            "03:25:45    ");
}

TEST(chrono_test, tm_align) {
  auto t = make_tm(1975, 12, 29, 12, 14, 16);
  EXPECT_EQ(fmt::format("{:%F %T}", t), "1975-12-29 12:14:16");
  EXPECT_EQ(fmt::format("{:30%F %T}", t), "1975-12-29 12:14:16           ");
  EXPECT_EQ(fmt::format("{:{}%F %T}", t, 30), "1975-12-29 12:14:16           ");
  EXPECT_EQ(fmt::format("{:<30%F %T}", t), "1975-12-29 12:14:16           ");
  EXPECT_EQ(fmt::format("{:^30%F %T}", t), "     1975-12-29 12:14:16      ");
  EXPECT_EQ(fmt::format("{:>30%F %T}", t), "           1975-12-29 12:14:16");

  EXPECT_EQ(fmt::format("{:*<30%F %T}", t), "1975-12-29 12:14:16***********");
  EXPECT_EQ(fmt::format("{:*^30%F %T}", t), "*****1975-12-29 12:14:16******");
  EXPECT_EQ(fmt::format("{:*>30%F %T}", t), "***********1975-12-29 12:14:16");
}

TEST(chrono_test, tp_align) {
  auto tp = std::chrono::time_point_cast<std::chrono::microseconds>(
      std::chrono::system_clock::from_time_t(0));
  EXPECT_EQ(fmt::format("{:%M:%S}", tp), "00:00.000000");
  EXPECT_EQ(fmt::format("{:15%M:%S}", tp), "00:00.000000   ");
  EXPECT_EQ(fmt::format("{:{}%M:%S}", tp, 15), "00:00.000000   ");
  EXPECT_EQ(fmt::format("{:<15%M:%S}", tp), "00:00.000000   ");
  EXPECT_EQ(fmt::format("{:^15%M:%S}", tp), " 00:00.000000  ");
  EXPECT_EQ(fmt::format("{:>15%M:%S}", tp), "   00:00.000000");

  EXPECT_EQ(fmt::format("{:*<15%M:%S}", tp), "00:00.000000***");
  EXPECT_EQ(fmt::format("{:*^15%M:%S}", tp), "*00:00.000000**");
  EXPECT_EQ(fmt::format("{:*>15%M:%S}", tp), "***00:00.000000");
}

TEST(chrono_test, format_specs) {
  EXPECT_EQ(fmt::format("{:%%}", std::chrono::seconds(0)), "%");
  EXPECT_EQ(fmt::format("{:%n}", std::chrono::seconds(0)), "\n");
  EXPECT_EQ(fmt::format("{:%t}", std::chrono::seconds(0)), "\t");
  EXPECT_EQ(fmt::format("{:%S}", std::chrono::seconds(0)), "00");
  EXPECT_EQ(fmt::format("{:%S}", std::chrono::seconds(60)), "00");
  EXPECT_EQ(fmt::format("{:%S}", std::chrono::seconds(42)), "42");
  EXPECT_EQ(fmt::format("{:%S}", std::chrono::milliseconds(1234)), "01.234");
  EXPECT_EQ(fmt::format("{:%M}", std::chrono::minutes(0)), "00");
  EXPECT_EQ(fmt::format("{:%M}", std::chrono::minutes(60)), "00");
  EXPECT_EQ(fmt::format("{:%M}", std::chrono::minutes(42)), "42");
  EXPECT_EQ(fmt::format("{:%M}", std::chrono::seconds(61)), "01");
  EXPECT_EQ(fmt::format("{:%H}", std::chrono::hours(0)), "00");
  EXPECT_EQ(fmt::format("{:%H}", std::chrono::hours(24)), "00");
  EXPECT_EQ(fmt::format("{:%H}", std::chrono::hours(14)), "14");
  EXPECT_EQ(fmt::format("{:%H}", std::chrono::minutes(61)), "01");
  EXPECT_EQ(fmt::format("{:%I}", std::chrono::hours(0)), "12");
  EXPECT_EQ(fmt::format("{:%I}", std::chrono::hours(12)), "12");
  EXPECT_EQ(fmt::format("{:%I}", std::chrono::hours(24)), "12");
  EXPECT_EQ(fmt::format("{:%I}", std::chrono::hours(4)), "04");
  EXPECT_EQ(fmt::format("{:%I}", std::chrono::hours(14)), "02");
  EXPECT_EQ(fmt::format("{:%j}", days(12)), "12");
  EXPECT_EQ(fmt::format("{:%j}", days(12345)), "12345");
  EXPECT_EQ(fmt::format("{:%j}", std::chrono::hours(12345 * 24 + 12)), "12345");
  EXPECT_EQ(fmt::format("{:%H:%M:%S}", std::chrono::seconds(12345)),
            "03:25:45");
  EXPECT_EQ(fmt::format("{:%R}", std::chrono::seconds(12345)), "03:25");
  EXPECT_EQ(fmt::format("{:%T}", std::chrono::seconds(12345)), "03:25:45");
  EXPECT_EQ(fmt::format("{:%Q}", std::chrono::seconds(12345)), "12345");
  EXPECT_EQ(fmt::format("{:%q}", std::chrono::seconds(12345)), "s");
}

TEST(chrono_test, invalid_specs) {
  auto sec = std::chrono::seconds(0);
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%a}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%A}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%c}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%x}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%Ex}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%X}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%EX}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%D}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%F}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%Ec}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%w}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%u}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%b}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%B}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%z}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%Z}"), sec), fmt::format_error,
                   "no date");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%Eq}"), sec), fmt::format_error,
                   "invalid format");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%Oq}"), sec), fmt::format_error,
                   "invalid format");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:abc}"), sec), fmt::format_error,
                   "invalid format");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:.2f}"), sec), fmt::format_error,
                   "invalid format");
}

auto format_tm(const std::tm& time, fmt::string_view spec,
               const std::locale& loc) -> std::string {
  auto& facet = std::use_facet<std::time_put<char>>(loc);
  std::ostringstream os;
  os.imbue(loc);
  facet.put(os, os, ' ', &time, spec.begin(), spec.end());
  return os.str();
}

TEST(chrono_test, locale) {
  auto loc = get_locale("ja_JP.utf8");
  if (loc == std::locale::classic()) return;
#define EXPECT_TIME(spec, time, duration)                     \
  {                                                           \
    auto jp_loc = std::locale("ja_JP.utf8");                  \
    EXPECT_EQ(format_tm(time, spec, jp_loc),                  \
              fmt::format(jp_loc, "{:L" spec "}", duration)); \
  }
  EXPECT_TIME("%OH", make_hour(14), std::chrono::hours(14));
  EXPECT_TIME("%OI", make_hour(14), std::chrono::hours(14));
  EXPECT_TIME("%OM", make_minute(42), std::chrono::minutes(42));
  EXPECT_TIME("%OS", make_second(42), std::chrono::seconds(42));
  auto time = make_tm();
  time.tm_hour = 3;
  time.tm_min = 25;
  time.tm_sec = 45;
  auto sec = std::chrono::seconds(12345);
  EXPECT_TIME("%r", time, sec);
  EXPECT_TIME("%p", time, sec);
}

using dms = std::chrono::duration<double, std::milli>;

TEST(chrono_test, format_default_fp) {
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<float>(1.234)), "1.234s");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<float, std::milli>(1.234)),
            "1.234ms");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<double>(1.234)), "1.234s");
  EXPECT_EQ(fmt::format("{}", dms(1.234)), "1.234ms");
}

TEST(chrono_test, format_precision) {
  EXPECT_THROW_MSG(
      (void)fmt::format(runtime("{:.2%Q}"), std::chrono::seconds(42)),
      fmt::format_error, "precision not allowed for this argument type");
  EXPECT_EQ(fmt::format("{:.0}", dms(1.234)), "1ms");
  EXPECT_EQ(fmt::format("{:.1}", dms(1.234)), "1.2ms");
  EXPECT_EQ(fmt::format("{:.{}}", dms(1.234), 2), "1.23ms");

  EXPECT_EQ(fmt::format("{:.0}", dms(12.56)), "13ms");
  EXPECT_EQ(fmt::format("{:.1}", dms(12.56)), "12.6ms");
  EXPECT_EQ(fmt::format("{:.2}", dms(12.56)), "12.56ms");
}

TEST(chrono_test, format_full_specs) {
  EXPECT_EQ(fmt::format("{:6.0}", dms(1.234)), "1ms   ");
  EXPECT_EQ(fmt::format("{:6.1}", dms(1.234)), "1.2ms ");
  EXPECT_EQ(fmt::format("{:>8.{}}", dms(1.234), 2), "  1.23ms");
  EXPECT_EQ(fmt::format("{:^{}.{}}", dms(1.234), 7, 1), " 1.2ms ");
  EXPECT_EQ(fmt::format("{0:^{2}.{1}}", dms(1.234), 2, 8), " 1.23ms ");
  EXPECT_EQ(fmt::format("{:=^{}.{}}", dms(1.234), 9, 3), "=1.234ms=");
  EXPECT_EQ(fmt::format("{:*^10.4}", dms(1.234)), "*1.2340ms*");

  EXPECT_EQ(fmt::format("{:6.0}", dms(12.56)), "13ms  ");
  EXPECT_EQ(fmt::format("{:>8.{}}", dms(12.56), 0), "    13ms");
  EXPECT_EQ(fmt::format("{:^{}.{}}", dms(12.56), 6, 0), " 13ms ");
  EXPECT_EQ(fmt::format("{0:^{2}.{1}}", dms(12.56), 0, 8), "  13ms  ");
  EXPECT_EQ(fmt::format("{:=^{}.{}}", dms(12.56), 9, 0), "==13ms===");
  EXPECT_EQ(fmt::format("{:*^10.0}", dms(12.56)), "***13ms***");
}

TEST(chrono_test, format_simple_q) {
  EXPECT_EQ(fmt::format("{:%Q %q}", std::chrono::duration<float>(1.234)),
            "1.234 s");
  EXPECT_EQ(
      fmt::format("{:%Q %q}", std::chrono::duration<float, std::milli>(1.234)),
      "1.234 ms");
  EXPECT_EQ(fmt::format("{:%Q %q}", std::chrono::duration<double>(1.234)),
            "1.234 s");
  EXPECT_EQ(fmt::format("{:%Q %q}", dms(1.234)), "1.234 ms");
}

TEST(chrono_test, format_precision_q) {
  EXPECT_THROW_MSG(
      (void)fmt::format(runtime("{:.2%Q %q}"), std::chrono::seconds(42)),
      fmt::format_error, "precision not allowed for this argument type");
  EXPECT_EQ(fmt::format("{:.1%Q %q}", dms(1.234)), "1.2 ms");
  EXPECT_EQ(fmt::format("{:.{}%Q %q}", dms(1.234), 2), "1.23 ms");
}

TEST(chrono_test, format_full_specs_q) {
  EXPECT_EQ(fmt::format("{:7.0%Q %q}", dms(1.234)), "1 ms   ");
  EXPECT_EQ(fmt::format("{:7.1%Q %q}", dms(1.234)), "1.2 ms ");
  EXPECT_EQ(fmt::format("{:>8.{}%Q %q}", dms(1.234), 2), " 1.23 ms");
  EXPECT_EQ(fmt::format("{:^{}.{}%Q %q}", dms(1.234), 8, 1), " 1.2 ms ");
  EXPECT_EQ(fmt::format("{0:^{2}.{1}%Q %q}", dms(1.234), 2, 9), " 1.23 ms ");
  EXPECT_EQ(fmt::format("{:=^{}.{}%Q %q}", dms(1.234), 10, 3), "=1.234 ms=");
  EXPECT_EQ(fmt::format("{:*^11.4%Q %q}", dms(1.234)), "*1.2340 ms*");

  EXPECT_EQ(fmt::format("{:7.0%Q %q}", dms(12.56)), "13 ms  ");
  EXPECT_EQ(fmt::format("{:>8.{}%Q %q}", dms(12.56), 0), "   13 ms");
  EXPECT_EQ(fmt::format("{:^{}.{}%Q %q}", dms(12.56), 8, 0), " 13 ms  ");
  EXPECT_EQ(fmt::format("{0:^{2}.{1}%Q %q}", dms(12.56), 0, 9), "  13 ms  ");
  EXPECT_EQ(fmt::format("{:=^{}.{}%Q %q}", dms(12.56), 9, 0), "==13 ms==");
  EXPECT_EQ(fmt::format("{:*^11.0%Q %q}", dms(12.56)), "***13 ms***");
}

TEST(chrono_test, invalid_width_id) {
  EXPECT_THROW((void)fmt::format(runtime("{:{o}"), std::chrono::seconds(0)),
               fmt::format_error);
}

TEST(chrono_test, invalid_colons) {
  EXPECT_THROW((void)fmt::format(runtime("{0}=:{0::"), std::chrono::seconds(0)),
               fmt::format_error);
}

TEST(chrono_test, negative_durations) {
  EXPECT_EQ(fmt::format("{:%Q}", std::chrono::seconds(-12345)), "-12345");
  EXPECT_EQ(fmt::format("{:%H:%M:%S}", std::chrono::seconds(-12345)),
            "-03:25:45");
  EXPECT_EQ(fmt::format("{:%M:%S}", std::chrono::duration<double>(-1)),
            "-00:01");
  EXPECT_EQ(fmt::format("{:%q}", std::chrono::seconds(-12345)), "s");
  EXPECT_EQ(fmt::format("{:%S}",
                        std::chrono::duration<signed char, std::milli>(-127)),
            "-00.127");
  auto min = std::numeric_limits<int>::min();
  EXPECT_EQ(fmt::format("{}", min),
            fmt::format("{:%Q}", std::chrono::duration<int>(min)));
}

TEST(chrono_test, special_durations) {
  EXPECT_EQ(fmt::format("{:%S}", std::chrono::duration<double>(1e20)), "40");
  auto nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(
      fmt::format("{:%I %H %M %S %R %r}", std::chrono::duration<double>(nan)),
      "nan nan nan nan nan:nan nan");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<float, std::exa>(1)),
            "1Es");
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<float, std::atto>(1)),
            "1as");
  EXPECT_EQ(fmt::format("{:%R}", std::chrono::duration<char, std::mega>{2}),
            "03:33");
  EXPECT_EQ(fmt::format("{:%T}", std::chrono::duration<char, std::mega>{2}),
            "03:33:20");
  EXPECT_EQ(
      fmt::format("{:.3%S}", std::chrono::duration<float, std::pico>(1.234e12)),
      "01.234");
}

TEST(chrono_test, unsigned_duration) {
  EXPECT_EQ(fmt::format("{}", std::chrono::duration<unsigned>(42)), "42s");
}

TEST(chrono_test, weekday) {
  auto loc = get_locale("es_ES.UTF-8");
  std::locale::global(loc);

  auto sat = fmt::weekday(6);

  auto tm = std::tm();
  tm.tm_wday = static_cast<int>(sat.c_encoding());

  EXPECT_EQ(fmt::format("{}", sat), "Sat");
  EXPECT_EQ(fmt::format("{:%a}", sat), "Sat");
  EXPECT_EQ(fmt::format("{:%A}", sat), "Saturday");
  EXPECT_EQ(fmt::format("{:%a}", tm), "Sat");

  if (loc != std::locale::classic()) {
    auto saturdays = std::vector<std::string>{"sáb", "sá.", "sáb."};
    EXPECT_THAT(saturdays, Contains(fmt::format(loc, "{:L}", sat)));
    EXPECT_THAT(saturdays, Contains(fmt::format(loc, "{:L%a}", sat)));
    EXPECT_THAT(saturdays, Contains(fmt::format(loc, "{:L%a}", tm)));
  }
}

TEST(chrono_test, cpp20_duration_subsecond_support) {
  using attoseconds = std::chrono::duration<long long, std::atto>;
  // Check that 18 digits of subsecond precision are supported.
  EXPECT_EQ(fmt::format("{:%S}", attoseconds{999999999999999999}),
            "00.999999999999999999");
  EXPECT_EQ(fmt::format("{:%S}", attoseconds{673231113420148734}),
            "00.673231113420148734");
  EXPECT_EQ(fmt::format("{:%S}", attoseconds{-673231113420148734}),
            "-00.673231113420148734");
  EXPECT_EQ(fmt::format("{:%S}", std::chrono::nanoseconds{13420148734}),
            "13.420148734");
  EXPECT_EQ(fmt::format("{:%S}", std::chrono::nanoseconds{-13420148734}),
            "-13.420148734");
  EXPECT_EQ(fmt::format("{:%S}", std::chrono::milliseconds{1234}), "01.234");
  // Check subsecond presision modifier.
  EXPECT_EQ(fmt::format("{:.6%S}", std::chrono::nanoseconds{1234}),
            "00.000001");
  EXPECT_EQ(fmt::format("{:.18%S}", std::chrono::nanoseconds{1234}),
            "00.000001234000000000");
  EXPECT_EQ(fmt::format("{:.{}%S}", std::chrono::nanoseconds{1234}, 6),
            "00.000001");
  EXPECT_EQ(fmt::format("{:.6%S}", std::chrono::milliseconds{1234}),
            "01.234000");
  EXPECT_EQ(fmt::format("{:.6%S}", std::chrono::milliseconds{-1234}),
            "-01.234000");
  EXPECT_EQ(fmt::format("{:.2%S}", std::chrono::milliseconds{12345}), "12.34");
  EXPECT_EQ(fmt::format("{:.2%S}", std::chrono::milliseconds{12375}), "12.37");
  EXPECT_EQ(fmt::format("{:.2%S}", std::chrono::milliseconds{-12375}),
            "-12.37");
  EXPECT_EQ(fmt::format("{:.0%S}", std::chrono::milliseconds{12054}), "12");
  EXPECT_EQ(fmt::format("{:.2%S}", std::chrono::milliseconds{99999}), "39.99");
  EXPECT_EQ(fmt::format("{:.2%S}", std::chrono::milliseconds{1000}), "01.00");
  EXPECT_EQ(fmt::format("{:.3%S}", std::chrono::milliseconds{1}), "00.001");
  EXPECT_EQ(fmt::format("{:.3%S}", std::chrono::seconds{1234}), "34.000");
  EXPECT_EQ(fmt::format("{:.3%S}", std::chrono::hours{1234}), "00.000");
  EXPECT_EQ(fmt::format("{:.5%S}", dms(1.234)), "00.00123");
  EXPECT_EQ(fmt::format("{:.8%S}", dms(1.234)), "00.00123400");
  {
    // Check that {:%H:%M:%S} is equivalent to {:%T}.
    auto dur = std::chrono::milliseconds{3601234};
    auto formatted_dur = fmt::format("{:%T}", dur);
    EXPECT_EQ(formatted_dur, "01:00:01.234");
    EXPECT_EQ(fmt::format("{:%H:%M:%S}", dur), formatted_dur);
    EXPECT_EQ(fmt::format("{:.6%H:%M:%S}", dur), "01:00:01.234000");
  }
  using nanoseconds_dbl = std::chrono::duration<double, std::nano>;
  EXPECT_EQ(fmt::format("{:%S}", nanoseconds_dbl(-123456789)), "-00.123456789");
  EXPECT_EQ(fmt::format("{:%S}", nanoseconds_dbl(9123456789)), "09.123456789");
  // Verify that only the seconds part is extracted and printed.
  EXPECT_EQ(fmt::format("{:%S}", nanoseconds_dbl(99123456789)), "39.123456789");
  EXPECT_EQ(fmt::format("{:%S}", nanoseconds_dbl(99123000000)), "39.123000000");
  {
    // Now the hour is printed, and we also test if negative doubles work.
    auto dur = nanoseconds_dbl{-99123456789};
    auto formatted_dur = fmt::format("{:%T}", dur);
    EXPECT_EQ(formatted_dur, "-00:01:39.123456789");
    EXPECT_EQ(fmt::format("{:%H:%M:%S}", dur), formatted_dur);
    EXPECT_EQ(fmt::format("{:.3%H:%M:%S}", dur), "-00:01:39.123");
  }
  // Check that durations with precision greater than std::chrono::seconds have
  // fixed precision, and print zeros even if there is no fractional part.
  EXPECT_EQ(fmt::format("{:%S}", std::chrono::microseconds(7000000)),
            "07.000000");
  EXPECT_EQ(fmt::format("{:%S}",
                        std::chrono::duration<long long, std::ratio<1, 3>>(1)),
            "00.333333");
  EXPECT_EQ(fmt::format("{:%S}",
                        std::chrono::duration<long long, std::ratio<1, 7>>(1)),
            "00.142857");

  EXPECT_EQ(
      fmt::format("{:%S}",
                  std::chrono::duration<signed char, std::ratio<1, 100>>(0x80)),
      "-01.28");

  EXPECT_EQ(
      fmt::format("{:%M:%S}",
                  std::chrono::duration<short, std::ratio<1, 100>>(0x8000)),
      "-05:27.68");

  // Check that floating point seconds with ratio<1,1> are printed.
  EXPECT_EQ(fmt::format("{:%S}", std::chrono::duration<double>(1.5)),
            "01.500000");
  EXPECT_EQ(fmt::format("{:%M:%S}", std::chrono::duration<double>(-61.25)),
            "-01:01.250000");
}

// Disable the utc_clock test for windows, as the icu.dll used for tzdb
// (time zone database) is not shipped with many windows versions.
#if FMT_USE_UTC_TIME && !defined(_WIN32)
TEST(chrono_test, utc_clock) {
  auto t1 = std::chrono::system_clock::now();
  auto t1_utc = std::chrono::utc_clock::from_sys(t1);
  EXPECT_EQ(fmt::format("{:%Y-%m-%d %H:%M:%S}", t1),
            fmt::format("{:%Y-%m-%d %H:%M:%S}", t1_utc));
}
#endif

TEST(chrono_test, timestamp_ratios) {
  auto t1 =
      sys_time<std::chrono::milliseconds>(std::chrono::milliseconds(67890));
  EXPECT_EQ(fmt::format("{:%M:%S}", t1), "01:07.890");

  auto t2 = sys_time<std::chrono::minutes>(std::chrono::minutes(7));
  EXPECT_EQ(fmt::format("{:%M:%S}", t2), "07:00");

  auto t3 = sys_time<std::chrono::duration<int, std::ratio<9>>>(
      std::chrono::duration<int, std::ratio<9>>(7));
  EXPECT_EQ(fmt::format("{:%M:%S}", t3), "01:03");

  auto t4 = sys_time<std::chrono::duration<int, std::ratio<63>>>(
      std::chrono::duration<int, std::ratio<63>>(1));
  EXPECT_EQ(fmt::format("{:%M:%S}", t4), "01:03");

  if (sizeof(time_t) > 4) {
    auto tp =
        sys_time<std::chrono::milliseconds>(std::chrono::seconds(32503680000));
    EXPECT_EQ(fmt::format("{:%Y-%m-%d}", tp), "3000-01-01");
  }

  if (FMT_SAFE_DURATION_CAST) {
    using years = std::chrono::duration<std::int64_t, std::ratio<31556952>>;
    auto tp = sys_time<years>(years(std::numeric_limits<std::int64_t>::max()));
    EXPECT_THROW_MSG((void)fmt::format("{:%Y-%m-%d}", tp), fmt::format_error,
                     "cannot format duration");
  }
}

TEST(chrono_test, timestamp_sub_seconds) {
  auto t1 = sys_time<std::chrono::duration<long long, std::ratio<1, 3>>>(
      std::chrono::duration<long long, std::ratio<1, 3>>(4));
  EXPECT_EQ(fmt::format("{:%S}", t1), "01.333333");

  auto t2 = sys_time<std::chrono::duration<double, std::ratio<1, 3>>>(
      std::chrono::duration<double, std::ratio<1, 3>>(4));
  EXPECT_EQ(fmt::format("{:%S}", t2), "01.333333");

  auto t3 = sys_time<std::chrono::seconds>(std::chrono::seconds(2));
  EXPECT_EQ(fmt::format("{:%S}", t3), "02");

  auto t4 = sys_time<std::chrono::duration<double>>(
      std::chrono::duration<double, std::ratio<1, 1>>(9.5));
  EXPECT_EQ(fmt::format("{:%S}", t4), "09.500000");

  auto t5 = sys_time<std::chrono::duration<double>>(
      std::chrono::duration<double, std::ratio<1, 1>>(9));
  EXPECT_EQ(fmt::format("{:%S}", t5), "09");

  auto t6 = sys_time<std::chrono::milliseconds>(std::chrono::seconds(1) +
                                                std::chrono::milliseconds(120));
  EXPECT_EQ(fmt::format("{:%S}", t6), "01.120");

  auto t7 =
      sys_time<std::chrono::microseconds>(std::chrono::microseconds(1234567));
  EXPECT_EQ(fmt::format("{:%S}", t7), "01.234567");

  auto t8 =
      sys_time<std::chrono::nanoseconds>(std::chrono::nanoseconds(123456789));
  EXPECT_EQ(fmt::format("{:%S}", t8), "00.123456789");
  EXPECT_EQ(fmt::format("{:%T}", t8), "00:00:00.123456789");

  auto t9 =
      sys_time<std::chrono::milliseconds>(std::chrono::milliseconds(2000));
  EXPECT_EQ(fmt::format("{:%S}", t9), "02.000");

  auto epoch = sys_time<std::chrono::milliseconds>();
  auto d = std::chrono::milliseconds(250);
  EXPECT_EQ(fmt::format("{:%S}", epoch - d), "59.750");
  EXPECT_EQ(fmt::format("{:%S}", epoch), "00.000");
  EXPECT_EQ(fmt::format("{:%S}", epoch + d), "00.250");
}

TEST(chrono_test, glibc_extensions) {
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%0}"), std::chrono::seconds()),
                   fmt::format_error, "invalid format");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%_}"), std::chrono::seconds()),
                   fmt::format_error, "invalid format");
  EXPECT_THROW_MSG((void)fmt::format(runtime("{:%-}"), std::chrono::seconds()),
                   fmt::format_error, "invalid format");

  {
    const auto d = std::chrono::hours(1) + std::chrono::minutes(2) +
                   std::chrono::seconds(3);

    EXPECT_EQ(fmt::format("{:%I,%H,%M,%S}", d), "01,01,02,03");
    EXPECT_EQ(fmt::format("{:%_I,%_H,%_M,%_S}", d), " 1, 1, 2, 3");
    EXPECT_EQ(fmt::format("{:%-I,%-H,%-M,%-S}", d), "1,1,2,3");
    EXPECT_EQ(fmt::format("{:%-I,%H,%M,%S}", d), "1,01,02,03");

    EXPECT_EQ(fmt::format("{:%OI,%OH,%OM,%OS}", d), "01,01,02,03");
    EXPECT_EQ(fmt::format("{:%_OI,%_OH,%_OM,%_OS}", d), " 1, 1, 2, 3");
    EXPECT_EQ(fmt::format("{:%-OI,%-OH,%-OM,%-OS}", d), "1,1,2,3");
  }

  {
    const auto tm = make_tm(1970, 1, 1, 1, 2, 3);
    EXPECT_EQ(fmt::format("{:%I,%H,%M,%S}", tm), "01,01,02,03");
    EXPECT_EQ(fmt::format("{:%_I,%_H,%_M,%_S}", tm), " 1, 1, 2, 3");
    EXPECT_EQ(fmt::format("{:%-I,%-H,%-M,%-S}", tm), "1,1,2,3");

    EXPECT_EQ(fmt::format("{:%OI,%OH,%OM,%OS}", tm), "01,01,02,03");
    EXPECT_EQ(fmt::format("{:%_OI,%_OH,%_OM,%_OS}", tm), " 1, 1, 2, 3");
    EXPECT_EQ(fmt::format("{:%-OI,%-OH,%-OM,%-OS}", tm), "1,1,2,3");
  }

  {
    const auto d = std::chrono::seconds(3) + std::chrono::milliseconds(140);
    EXPECT_EQ(fmt::format("{:%S}", d), "03.140");
    EXPECT_EQ(fmt::format("{:%_S}", d), " 3.140");
    EXPECT_EQ(fmt::format("{:%-S}", d), "3.140");
  }

  {
    auto d = std::chrono::duration<double>(3.14);
    EXPECT_EQ(fmt::format("{:%S}", d), "03.140000");
    EXPECT_EQ(fmt::format("{:%_S}", d), " 3.140000");
    EXPECT_EQ(fmt::format("{:%-S}", d), "3.140000");
  }

  {
    auto t = std::tm();
    t.tm_yday = 7;
    EXPECT_EQ(fmt::format("{:%U,%W,%V}", t), "02,01,01");
    EXPECT_EQ(fmt::format("{:%_U,%_W,%_V}", t), " 2, 1, 1");
    EXPECT_EQ(fmt::format("{:%-U,%-W,%-V}", t), "2,1,1");

    EXPECT_EQ(fmt::format("{:%j}", t), "008");
    EXPECT_EQ(fmt::format("{:%_j}", t), "  8");
    EXPECT_EQ(fmt::format("{:%-j}", t), "8");
  }

  {
    auto t = std::tm();
    t.tm_mday = 7;
    EXPECT_EQ(fmt::format("{:%d}", t), "07");
    EXPECT_EQ(fmt::format("{:%_d}", t), " 7");
    EXPECT_EQ(fmt::format("{:%-d}", t), "7");

    EXPECT_EQ(fmt::format("{:%e}", t), " 7");
  }

  {
    auto t = std::tm();
    t.tm_year = 7 - 1900;
    EXPECT_EQ(fmt::format("{:%Y}", t), "0007");
    EXPECT_EQ(fmt::format("{:%_Y}", t), "   7");
    EXPECT_EQ(fmt::format("{:%-Y}", t), "7");
  }

  {
    auto t = std::tm();
    t.tm_year = -5 - 1900;
    EXPECT_EQ(fmt::format("{:%Y}", t), "-005");
    EXPECT_EQ(fmt::format("{:%_Y}", t), "  -5");
    EXPECT_EQ(fmt::format("{:%-Y}", t), "-5");
  }

  {
    auto t = std::tm();
    t.tm_mon = 7 - 1;
    EXPECT_EQ(fmt::format("{:%m}", t), "07");
    EXPECT_EQ(fmt::format("{:%_m}", t), " 7");
    EXPECT_EQ(fmt::format("{:%-m}", t), "7");
  }
}

TEST(chrono_test, out_of_range) {
  auto d = std::chrono::duration<unsigned long, std::giga>(538976288);
  EXPECT_THROW((void)fmt::format("{:%j}", d), fmt::format_error);
}

TEST(chrono_test, year_month_day) {
  auto loc = get_locale("es_ES.UTF-8");
  std::locale::global(loc);

  auto year = fmt::year(2024);
  auto month = fmt::month(1);
  auto day = fmt::day(1);
  auto ymd = fmt::year_month_day(year, month, day);

  EXPECT_EQ(fmt::format("{}", year), "2024");
  EXPECT_EQ(fmt::format("{:%Y}", year), "2024");
  EXPECT_EQ(fmt::format("{:%y}", year), "24");

  EXPECT_EQ(fmt::format("{}", month), "Jan");
  EXPECT_EQ(fmt::format("{:%m}", month), "01");
  EXPECT_EQ(fmt::format("{:%b}", month), "Jan");
  EXPECT_EQ(fmt::format("{:%B}", month), "January");

  EXPECT_EQ(fmt::format("{}", day), "01");
  EXPECT_EQ(fmt::format("{:%d}", day), "01");

  EXPECT_EQ(fmt::format("{}", ymd), "2024-01-01");
  EXPECT_EQ(fmt::format("{:%Y-%m-%d}", ymd), "2024-01-01");
  EXPECT_EQ(fmt::format("{:%Y-%b-%d}", ymd), "2024-Jan-01");
  EXPECT_EQ(fmt::format("{:%Y-%B-%d}", ymd), "2024-January-01");

  if (loc != std::locale::classic()) {
    auto months = std::vector<std::string>{"ene.", "ene"};
    EXPECT_THAT(months, Contains(fmt::format(loc, "{:L}", month)));
    EXPECT_THAT(months, Contains(fmt::format(loc, "{:L%b}", month)));
  }
}
