package config

import (
	"os"
)

// Config holds the application configuration
type Config struct {
	Endpoint      string
	TokenizerPath string
	Port          string
	LogDir        string
	LogLevel      string
}

// Load loads configuration from environment variables with defaults
func Load() *Config {
	// Get tokenizer path from environment or use default
	tokenizerPath := os.Getenv("SGL_TOKENIZER_PATH")
	if tokenizerPath == "" {
		tokenizerPath = "../tokenizer"
	}

	// Get endpoint from environment or use default
	endpoint := os.Getenv("SGL_GRPC_ENDPOINT")
	if endpoint == "" {
		endpoint = "grpc://localhost:20000"
	}

	// Get port from environment or use default
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Get log directory from environment or use default
	logDir := os.Getenv("LOG_DIR")
	if logDir == "" {
		logDir = "./logs"
	}

	// Get log level from environment or use default
	logLevel := os.Getenv("LOG_LEVEL")
	if logLevel == "" {
		logLevel = "info"
	}

	return &Config{
		Endpoint:      endpoint,
		TokenizerPath: tokenizerPath,
		Port:          port,
		LogDir:        logDir,
		LogLevel:      logLevel,
	}
}
