package sglang_tests

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/bazelbuild/bazel-gazelle/config"
	"github.com/bazelbuild/bazel-gazelle/label"
	"github.com/bazelbuild/bazel-gazelle/language"
	"github.com/bazelbuild/bazel-gazelle/repo"
	"github.com/bazelbuild/bazel-gazelle/resolve"
	"github.com/bazelbuild/bazel-gazelle/rule"
)

var registerCallRE = regexp.MustCompile(`(?m)^register_(cpu|cuda|amd|npu)_ci\s*\(`)

type sglangTestsLang struct {
	language.BaseLang
}

type registration struct {
	backend  string
	suite    string
	estTime  int
	nightly  bool
	disabled bool
}

func NewLanguage() language.Language {
	return &sglangTestsLang{}
}

func (l *sglangTestsLang) Name() string {
	return "sglang_tests"
}

func (l *sglangTestsLang) RegisterFlags(fs *flag.FlagSet, cmd string, c *config.Config) {}

func (l *sglangTestsLang) CheckFlags(fs *flag.FlagSet, c *config.Config) error {
	return nil
}

func (l *sglangTestsLang) KnownDirectives() []string {
	return nil
}

func (l *sglangTestsLang) Configure(c *config.Config, rel string, f *rule.File) {}

func (l *sglangTestsLang) Kinds() map[string]rule.KindInfo {
	info := rule.KindInfo{
		NonEmptyAttrs: map[string]bool{
			"est_time": true,
			"srcs":     true,
			"suite":    true,
		},
		MergeableAttrs: map[string]bool{
			"arch":     true,
			"data":     true,
			"deps":     true,
			"env":      true,
			"est_time": true,
			"main":     true,
			"nightly":  true,
			"srcs":     true,
			"suite":    true,
			"tags":     true,
		},
		ResolveAttrs: map[string]bool{"deps": true},
	}
	return map[string]rule.KindInfo{
		"sgl_amd_test":  info,
		"sgl_cpu_test":  info,
		"sgl_cuda_test": info,
		"sgl_npu_test":  info,
	}
}

func (l *sglangTestsLang) Loads() []rule.LoadInfo {
	return []rule.LoadInfo{sglangLoadInfo("//tools/bazel:sgl_defs.bzl")}
}

func (l *sglangTestsLang) ApparentLoads(moduleToApparentName func(string) string) []rule.LoadInfo {
	return l.Loads()
}

func (l *sglangTestsLang) Imports(c *config.Config, r *rule.Rule, f *rule.File) []resolve.ImportSpec {
	return nil
}

func (l *sglangTestsLang) Embeds(r *rule.Rule, from label.Label) []label.Label {
	return nil
}

func (l *sglangTestsLang) Resolve(c *config.Config, ix *resolve.RuleIndex, rc *repo.RemoteCache, r *rule.Rule, imports interface{}, from label.Label) {
}

func (l *sglangTestsLang) GenerateRules(args language.GenerateArgs) language.GenerateResult {
	if !strings.HasPrefix(args.Rel, "test/registered") {
		return language.GenerateResult{}
	}

	testsByFile := make(map[string][]registration)
	for _, filename := range args.RegularFiles {
		if !isPythonTestFile(filename) {
			continue
		}
		regs, err := parseRegistrations(filepath.Join(args.Dir, filename))
		if err != nil {
			log.Printf("%s/%s: %v", args.Rel, filename, err)
			continue
		}
		if len(regs) > 0 {
			testsByFile[filename] = regs
		}
	}
	if len(testsByFile) == 0 {
		return language.GenerateResult{}
	}

	pyTestsBySrc := pyTestsBySingleSrc(args.OtherGen)
	var generated []*rule.Rule
	var imports []interface{}
	for filename, regs := range testsByFile {
		enabled := enabledRegistrations(regs)
		if len(enabled) == 0 {
			continue
		}
		for i, reg := range enabled {
			var r *rule.Rule
			if i == 0 {
				r = pyTestsBySrc[filename]
			}
			if r == nil {
				r = rule.NewRule(kindForBackend(reg.backend), targetName(filename, reg, len(enabled) > 1))
				generated = append(generated, r)
				imports = append(imports, nil)
			}
			applyRegistration(r, filename, reg, len(enabled) > 1)
		}
	}

	return language.GenerateResult{
		Gen:     generated,
		Imports: imports,
	}
}

func sglangLoadInfo(name string) rule.LoadInfo {
	return rule.LoadInfo{
		Name: name,
		Symbols: []string{
			"sgl_amd_test",
			"sgl_cpu_test",
			"sgl_cuda_test",
			"sgl_npu_test",
		},
	}
}

func isPythonTestFile(filename string) bool {
	return strings.HasPrefix(filename, "test_") && strings.HasSuffix(filename, ".py")
}

func pyTestsBySingleSrc(rules []*rule.Rule) map[string]*rule.Rule {
	result := make(map[string]*rule.Rule)
	for _, r := range rules {
		if r.Kind() != "py_test" {
			continue
		}
		srcs := r.AttrStrings("srcs")
		if len(srcs) == 1 {
			result[srcs[0]] = r
		}
	}
	return result
}

func enabledRegistrations(regs []registration) []registration {
	var enabled []registration
	for _, reg := range regs {
		if !reg.disabled {
			enabled = append(enabled, reg)
		}
	}
	return enabled
}

func applyRegistration(r *rule.Rule, filename string, reg registration, needsSuffix bool) {
	r.SetKind(kindForBackend(reg.backend))
	r.SetName(targetName(filename, reg, needsSuffix))
	r.SetAttr("srcs", []string{filename})
	r.SetAttr("suite", reg.suite)
	r.SetAttr("est_time", reg.estTime)
	if reg.nightly {
		r.SetAttr("nightly", true)
	} else {
		r.DelAttr("nightly")
	}
}

func kindForBackend(backend string) string {
	switch backend {
	case "amd":
		return "sgl_amd_test"
	case "cpu":
		return "sgl_cpu_test"
	case "cuda":
		return "sgl_cuda_test"
	case "npu":
		return "sgl_npu_test"
	default:
		panic(fmt.Sprintf("unknown backend %q", backend))
	}
}

func targetName(filename string, reg registration, needsSuffix bool) string {
	base := strings.TrimSuffix(filename, ".py")
	if !needsSuffix {
		return base
	}
	name := base + "_" + reg.backend + "_" + sanitize(reg.suite)
	if reg.nightly {
		name += "_nightly"
	}
	return name
}

func sanitize(value string) string {
	var b strings.Builder
	lastUnderscore := false
	for _, r := range value {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(unicode.ToLower(r))
			lastUnderscore = false
			continue
		}
		if !lastUnderscore {
			b.WriteByte('_')
			lastUnderscore = true
		}
	}
	return strings.Trim(b.String(), "_")
}

func parseRegistrations(path string) ([]registration, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	text := string(content)
	matches := registerCallRE.FindAllStringSubmatchIndex(text, -1)
	var regs []registration
	for _, match := range matches {
		backend := text[match[2]:match[3]]
		openParen := match[1] - 1
		closeParen, err := findClosingParen(text, openParen)
		if err != nil {
			return nil, err
		}
		reg, err := parseCallArgs(backend, text[openParen+1:closeParen])
		if err != nil {
			return nil, err
		}
		regs = append(regs, reg)
	}
	return regs, nil
}

func findClosingParen(text string, open int) (int, error) {
	depth := 0
	var quote byte
	escaped := false
	for i := open; i < len(text); i++ {
		ch := text[i]
		if quote != 0 {
			if escaped {
				escaped = false
			} else if ch == '\\' {
				escaped = true
			} else if ch == quote {
				quote = 0
			}
			continue
		}
		switch ch {
		case '\'', '"':
			quote = ch
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return i, nil
			}
		}
	}
	return 0, fmt.Errorf("unterminated register_*_ci call")
}

func parseCallArgs(backend, argsText string) (registration, error) {
	args := map[string]string{}
	positional := splitTopLevel(argsText)
	order := []string{"est_time", "suite", "nightly", "disabled"}
	pos := 0
	for _, raw := range positional {
		arg := strings.TrimSpace(raw)
		if arg == "" {
			continue
		}
		if key, value, ok := splitKeyword(arg); ok {
			args[key] = strings.TrimSpace(value)
			continue
		}
		if pos >= len(order) {
			return registration{}, fmt.Errorf("too many positional arguments")
		}
		args[order[pos]] = arg
		pos++
	}

	estText, ok := args["est_time"]
	if !ok {
		return registration{}, fmt.Errorf("missing est_time")
	}
	suiteText, ok := args["suite"]
	if !ok {
		return registration{}, fmt.Errorf("missing suite")
	}

	estTime, err := parseEstTime(estText)
	if err != nil {
		return registration{}, fmt.Errorf("invalid est_time: %w", err)
	}
	suite, err := parseString(suiteText)
	if err != nil {
		return registration{}, fmt.Errorf("invalid suite: %w", err)
	}
	nightly := false
	if nightlyText, ok := args["nightly"]; ok {
		nightly, err = parseBool(nightlyText)
		if err != nil {
			return registration{}, fmt.Errorf("invalid nightly: %w", err)
		}
	}
	disabled := false
	if disabledText, ok := args["disabled"]; ok {
		disabled = strings.TrimSpace(disabledText) != "None"
	}

	return registration{
		backend:  backend,
		suite:    suite,
		estTime:  estTime,
		nightly:  nightly,
		disabled: disabled,
	}, nil
}

func splitTopLevel(text string) []string {
	var parts []string
	start := 0
	depth := 0
	var quote byte
	escaped := false
	for i := 0; i < len(text); i++ {
		ch := text[i]
		if quote != 0 {
			if escaped {
				escaped = false
			} else if ch == '\\' {
				escaped = true
			} else if ch == quote {
				quote = 0
			}
			continue
		}
		switch ch {
		case '\'', '"':
			quote = ch
		case '(', '[', '{':
			depth++
		case ')', ']', '}':
			depth--
		case ',':
			if depth == 0 {
				parts = append(parts, text[start:i])
				start = i + 1
			}
		}
	}
	parts = append(parts, text[start:])
	return parts
}

func splitKeyword(text string) (string, string, bool) {
	var quote byte
	escaped := false
	for i := 0; i < len(text); i++ {
		ch := text[i]
		if quote != 0 {
			if escaped {
				escaped = false
			} else if ch == '\\' {
				escaped = true
			} else if ch == quote {
				quote = 0
			}
			continue
		}
		if ch == '\'' || ch == '"' {
			quote = ch
			continue
		}
		if ch == '=' {
			return strings.TrimSpace(text[:i]), text[i+1:], true
		}
	}
	return "", "", false
}

func parseEstTime(text string) (int, error) {
	value, err := strconv.ParseFloat(strings.TrimSpace(text), 64)
	if err != nil {
		return 0, err
	}
	return int(math.Ceil(value)), nil
}

func parseString(text string) (string, error) {
	trimmed := strings.TrimSpace(text)
	if len(trimmed) < 2 {
		return "", fmt.Errorf("expected string literal")
	}
	if strings.HasPrefix(trimmed, "r") || strings.HasPrefix(trimmed, "R") {
		trimmed = trimmed[1:]
	}
	value, err := strconv.Unquote(trimmed)
	if err != nil {
		return "", err
	}
	return value, nil
}

func parseBool(text string) (bool, error) {
	switch strings.TrimSpace(text) {
	case "True":
		return true, nil
	case "False":
		return false, nil
	default:
		return false, fmt.Errorf("expected True or False")
	}
}
