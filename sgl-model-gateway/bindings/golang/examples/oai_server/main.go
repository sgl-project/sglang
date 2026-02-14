// OpenAI-compatible chat server using SGLang Go SDK and fasthttp framework
package main

import (
	"fmt"
	"net/http"
	"os"

	_ "net/http/pprof" // Enable pprof endpoints

	"github.com/valyala/fasthttp"
	"go.uber.org/zap"

	"oai_server/config"
	"oai_server/handlers"
	"oai_server/logger"
	"oai_server/service"
)

// Version information (set at build time via ldflags)
var (
	Version   = "dev"
	BuildTime = "unknown"
	GitCommit = "unknown"
)

func main() {
	// Load configuration
	cfg := config.Load()

	// Initialize logger
	appLogger, err := logger.Init(cfg.LogDir, cfg.LogLevel)
	if err != nil {
		panic(fmt.Sprintf("Failed to initialize logger: %v", err))
	}
	defer appLogger.Sync()

	appLogger.Info("Starting OpenAI-compatible server",
		zap.String("endpoint", cfg.Endpoint),
		zap.String("tokenizer", cfg.TokenizerPath),
		zap.String("port", cfg.Port),
	)

	// Initialize SGLang service
	sglangService, err := service.NewSGLangService(cfg.Endpoint, cfg.TokenizerPath)
	if err != nil {
		appLogger.Fatal("Failed to create SGLang client", zap.Error(err))
	}
	defer sglangService.Close()

	appLogger.Info("SGLang client created successfully")

	// Enable pprof if requested
	if os.Getenv("PPROF_ENABLED") == "true" {
		pprofPort := os.Getenv("PPROF_PORT")
		if pprofPort == "" {
			pprofPort = "6060"
		}
		go func() {
			pprofAddr := ":" + pprofPort
			appLogger.Info("Starting pprof server", zap.String("address", pprofAddr))
			if err := http.ListenAndServe(pprofAddr, nil); err != nil {
				appLogger.Error("pprof server failed", zap.Error(err))
			}
		}()
		appLogger.Info("pprof enabled", zap.String("port", pprofPort), zap.String("endpoint", fmt.Sprintf("http://localhost:%s/debug/pprof/", pprofPort)))
	}

	// Initialize handlers
	healthHandler := handlers.NewHealthHandler(appLogger)
	modelsHandler := handlers.NewModelsHandler(appLogger, cfg.TokenizerPath)
	chatHandler := handlers.NewChatHandler(appLogger, sglangService)

	// Setup fasthttp router
	router := func(ctx *fasthttp.RequestCtx) {
		path := string(ctx.Path())
		method := string(ctx.Method())

		switch {
		case method == "GET" && path == "/health":
			healthHandler.Check(ctx)
		case method == "GET" && path == "/v1/models":
			modelsHandler.List(ctx)
		case method == "GET" && path == "/get_model_info":
			modelsHandler.GetModelInfo(ctx)
		case method == "POST" && path == "/v1/chat/completions":
			chatHandler.HandleChatCompletion(ctx)
		case (method == "POST" || method == "PUT") && path == "/generate":
			chatHandler.HandleGenerate(ctx)
		default:
			ctx.Error("Not Found", fasthttp.StatusNotFound)
		}
	}

	// Start server
	serverAddr := ":" + cfg.Port
	baseURL := fmt.Sprintf("http://localhost:%s", cfg.Port)

	appLogger.Info("Server starting",
		zap.String("address", serverAddr),
		zap.String("base_url", baseURL),
	)

	// Print available HTTP endpoints (similar to FastAPI startup)
	appLogger.Info("Available HTTP endpoints:")
	appLogger.Info(fmt.Sprintf("  GET  %s/health", baseURL))
	appLogger.Info(fmt.Sprintf("  GET  %s/v1/models", baseURL))
	appLogger.Info(fmt.Sprintf("  GET  %s/get_model_info", baseURL))
	appLogger.Info(fmt.Sprintf("  POST %s/v1/chat/completions", baseURL))
	appLogger.Info(fmt.Sprintf("  POST %s/generate", baseURL))
	appLogger.Info(fmt.Sprintf("Application startup complete. Listening on %s", baseURL))

	if err := fasthttp.ListenAndServe(serverAddr, router); err != nil {
		appLogger.Fatal("Server failed", zap.Error(err))
	}
}
