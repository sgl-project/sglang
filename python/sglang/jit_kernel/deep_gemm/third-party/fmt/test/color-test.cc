// Formatting library for C++ - color tests
//
// Copyright (c) 2012 - present, Victor Zverovich
// All rights reserved.
//
// For the license information refer to format.h.

#include "fmt/color.h"

#include <iterator>  // std::back_inserter

#include "gtest-extra.h"  // EXPECT_WRITE, EXPECT_THROW_MSG

TEST(color_test, text_style) {
  EXPECT_FALSE(fmt::text_style().has_foreground());
  EXPECT_FALSE(fmt::text_style().has_background());
  EXPECT_FALSE(fmt::text_style().has_emphasis());

  EXPECT_TRUE(fg(fmt::rgb(0)).has_foreground());
  EXPECT_FALSE(fg(fmt::rgb(0)).has_background());
  EXPECT_FALSE(fg(fmt::rgb(0)).has_emphasis());
  EXPECT_TRUE(bg(fmt::rgb(0)).has_background());
  EXPECT_FALSE(bg(fmt::rgb(0)).has_foreground());
  EXPECT_FALSE(bg(fmt::rgb(0)).has_emphasis());

  EXPECT_TRUE(
      (fg(fmt::rgb(0xFFFFFF)) | bg(fmt::rgb(0xFFFFFF))).has_foreground());
  EXPECT_TRUE(
      (fg(fmt::rgb(0xFFFFFF)) | bg(fmt::rgb(0xFFFFFF))).has_background());
  EXPECT_FALSE(
      (fg(fmt::rgb(0xFFFFFF)) | bg(fmt::rgb(0xFFFFFF))).has_emphasis());

  EXPECT_EQ(fg(fmt::rgb(0x000000)) | fg(fmt::rgb(0x000000)),
            fg(fmt::rgb(0x000000)));
  EXPECT_EQ(fg(fmt::rgb(0x00000F)) | fg(fmt::rgb(0x00000F)),
            fg(fmt::rgb(0x00000F)));
  EXPECT_EQ(fg(fmt::rgb(0xC0F000)) | fg(fmt::rgb(0x000FEE)),
            fg(fmt::rgb(0xC0FFEE)));

  EXPECT_THROW_MSG(
      fg(fmt::terminal_color::black) | fg(fmt::terminal_color::black),
      fmt::format_error, "can't OR a terminal color");
  EXPECT_THROW_MSG(
      fg(fmt::terminal_color::black) | fg(fmt::terminal_color::white),
      fmt::format_error, "can't OR a terminal color");
  EXPECT_THROW_MSG(
      bg(fmt::terminal_color::black) | bg(fmt::terminal_color::black),
      fmt::format_error, "can't OR a terminal color");
  EXPECT_THROW_MSG(
      bg(fmt::terminal_color::black) | bg(fmt::terminal_color::white),
      fmt::format_error, "can't OR a terminal color");
  EXPECT_THROW_MSG(fg(fmt::terminal_color::black) | fg(fmt::color::black),
                   fmt::format_error, "can't OR a terminal color");
  EXPECT_THROW_MSG(bg(fmt::terminal_color::black) | bg(fmt::color::black),
                   fmt::format_error, "can't OR a terminal color");

  EXPECT_NO_THROW(fg(fmt::terminal_color::white) |
                  bg(fmt::terminal_color::white));
  EXPECT_NO_THROW(fg(fmt::terminal_color::white) | bg(fmt::rgb(0xFFFFFF)));
  EXPECT_NO_THROW(fg(fmt::terminal_color::white) | fmt::text_style());
  EXPECT_NO_THROW(bg(fmt::terminal_color::white) | fmt::text_style());
}

TEST(color_test, format) {
  EXPECT_EQ(fmt::format(fmt::text_style(), "no style"), "no style");
  EXPECT_EQ(fmt::format(fg(fmt::rgb(255, 20, 30)), "rgb(255,20,30)"),
            "\x1b[38;2;255;020;030mrgb(255,20,30)\x1b[0m");
  EXPECT_EQ(fmt::format(fg(fmt::rgb(255, 0, 0)) | fg(fmt::rgb(0, 20, 30)),
                        "rgb(255,20,30)"),
            "\x1b[38;2;255;020;030mrgb(255,20,30)\x1b[0m");
  EXPECT_EQ(
      fmt::format(fg(fmt::rgb(0, 0, 0)) | fg(fmt::rgb(0, 0, 0)), "rgb(0,0,0)"),
      "\x1b[38;2;000;000;000mrgb(0,0,0)\x1b[0m");
  EXPECT_EQ(fmt::format(fg(fmt::color::blue), "blue"),
            "\x1b[38;2;000;000;255mblue\x1b[0m");
  EXPECT_EQ(
      fmt::format(fg(fmt::color::blue) | bg(fmt::color::red), "two color"),
      "\x1b[38;2;000;000;255m\x1b[48;2;255;000;000mtwo color\x1b[0m");
  EXPECT_EQ(fmt::format(fmt::emphasis::bold, "bold"), "\x1b[1mbold\x1b[0m");
  EXPECT_EQ(fmt::format(fmt::emphasis::faint, "faint"), "\x1b[2mfaint\x1b[0m");
  EXPECT_EQ(fmt::format(fmt::emphasis::italic, "italic"),
            "\x1b[3mitalic\x1b[0m");
  EXPECT_EQ(fmt::format(fmt::emphasis::underline, "underline"),
            "\x1b[4munderline\x1b[0m");
  EXPECT_EQ(fmt::format(fmt::emphasis::blink, "blink"), "\x1b[5mblink\x1b[0m");
  EXPECT_EQ(fmt::format(fmt::emphasis::reverse, "reverse"),
            "\x1b[7mreverse\x1b[0m");
  EXPECT_EQ(fmt::format(fmt::emphasis::conceal, "conceal"),
            "\x1b[8mconceal\x1b[0m");
  EXPECT_EQ(fmt::format(fmt::emphasis::strikethrough, "strikethrough"),
            "\x1b[9mstrikethrough\x1b[0m");
  EXPECT_EQ(
      fmt::format(fg(fmt::color::blue) | fmt::emphasis::bold, "blue/bold"),
      "\x1b[1m\x1b[38;2;000;000;255mblue/bold\x1b[0m");
  EXPECT_EQ(fmt::format(fmt::emphasis::bold, "bold error"),
            "\x1b[1mbold error\x1b[0m");
  EXPECT_EQ(fmt::format(fg(fmt::color::blue), "blue log"),
            "\x1b[38;2;000;000;255mblue log\x1b[0m");
  EXPECT_EQ(fmt::format(fmt::text_style(), "hi"), "hi");
  EXPECT_EQ(fmt::format(fg(fmt::terminal_color::red), "tred"),
            "\x1b[31mtred\x1b[0m");
  EXPECT_EQ(fmt::format(bg(fmt::terminal_color::cyan), "tcyan"),
            "\x1b[46mtcyan\x1b[0m");
  EXPECT_EQ(fmt::format(fg(fmt::terminal_color::bright_green), "tbgreen"),
            "\x1b[92mtbgreen\x1b[0m");
  EXPECT_EQ(fmt::format(bg(fmt::terminal_color::bright_magenta), "tbmagenta"),
            "\x1b[105mtbmagenta\x1b[0m");
  EXPECT_EQ(fmt::format(fg(fmt::terminal_color::red), "{}", "foo"),
            "\x1b[31mfoo\x1b[0m");
  EXPECT_EQ(fmt::format("{}{}", fmt::styled("red", fg(fmt::color::red)),
                        fmt::styled("bold", fmt::emphasis::bold)),
            "\x1b[38;2;255;000;000mred\x1b[0m\x1b[1mbold\x1b[0m");
  EXPECT_EQ(fmt::format("{}", fmt::styled("bar", fg(fmt::color::blue) |
                                                     fmt::emphasis::underline)),
            "\x1b[4m\x1b[38;2;000;000;255mbar\x1b[0m");
}

TEST(color_test, format_to) {
  auto out = std::string();
  fmt::format_to(std::back_inserter(out), fg(fmt::rgb(255, 20, 30)),
                 "rgb(255,20,30){}{}{}", 1, 2, 3);
  EXPECT_EQ(fmt::to_string(out),
            "\x1b[38;2;255;020;030mrgb(255,20,30)123\x1b[0m");
}

TEST(color_test, print) {
  EXPECT_WRITE(stdout, fmt::print(fg(fmt::rgb(255, 20, 30)), "rgb(255,20,30)"),
               "\x1b[38;2;255;020;030mrgb(255,20,30)\x1b[0m");
}
