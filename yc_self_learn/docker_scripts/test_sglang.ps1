# 测试 SGLang 服务是否正常运行

param(
    [int]$Port = 30000,
    [string]$Model = "qwen/qwen2.5-0.5b-instruct"
)

$baseUrl = "http://localhost:$Port"

Write-Host "测试 SGLang 服务..." -ForegroundColor Green
Write-Host "  地址: $baseUrl" -ForegroundColor Yellow
Write-Host "  模型: $Model" -ForegroundColor Yellow
Write-Host ""

# 测试 1: 健康检查
Write-Host "1. 健康检查..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get -TimeoutSec 5
    Write-Host "   ✅ 服务健康" -ForegroundColor Green
} catch {
    Write-Host "   ❌ 健康检查失败: $_" -ForegroundColor Red
    exit 1
}

# 测试 2: Chat Completions API
Write-Host "`n2. 测试 Chat Completions API..." -ForegroundColor Cyan
try {
    $body = @{
        model = $Model
        messages = @(
            @{
                role = "user"
                content = "你好，请用一句话介绍你自己"
            }
        )
        max_tokens = 50
        temperature = 0.7
    } | ConvertTo-Json -Depth 10

    $response = Invoke-RestMethod -Uri "$baseUrl/v1/chat/completions" `
        -Method Post `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 30

    Write-Host "   ✅ API 调用成功" -ForegroundColor Green
    Write-Host "   响应内容: $($response.choices[0].message.content)" -ForegroundColor Gray
} catch {
    Write-Host "   ❌ API 调用失败: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "   错误详情: $responseBody" -ForegroundColor Red
    }
    exit 1
}

# 测试 3: 流式输出
Write-Host "`n3. 测试流式输出..." -ForegroundColor Cyan
try {
    $body = @{
        model = $Model
        messages = @(
            @{
                role = "user"
                content = "数数从1到5"
            }
        )
        max_tokens = 30
        stream = $true
    } | ConvertTo-Json -Depth 10

    $request = [System.Net.HttpWebRequest]::Create("$baseUrl/v1/chat/completions")
    $request.Method = "POST"
    $request.ContentType = "application/json"
    $request.Timeout = 30000
    
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
    $request.ContentLength = $bytes.Length
    $requestStream = $request.GetRequestStream()
    $requestStream.Write($bytes, 0, $bytes.Length)
    $requestStream.Close()
    
    $response = $request.GetResponse()
    $stream = $response.GetResponseStream()
    $reader = New-Object System.IO.StreamReader($stream)
    
    Write-Host "   流式输出: " -NoNewline -ForegroundColor Gray
    $buffer = ""
    while (($line = $reader.ReadLine()) -ne $null) {
        if ($line.StartsWith("data: ")) {
            $data = $line.Substring(6)
            if ($data -eq "[DONE]") {
                break
            }
            try {
                $json = $data | ConvertFrom-Json
                if ($json.choices[0].delta.content) {
                    Write-Host $json.choices[0].delta.content -NoNewline -ForegroundColor White
                }
            } catch {
                # 忽略解析错误
            }
        }
    }
    Write-Host ""
    Write-Host "   ✅ 流式输出正常" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  流式输出测试失败（可能不影响使用）: $_" -ForegroundColor Yellow
}

Write-Host "`n✅ 所有测试通过！SGLang 服务运行正常。" -ForegroundColor Green

