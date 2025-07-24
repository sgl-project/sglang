import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

HEADERS = {"Content-Type": "application/json"}

def send_request(request_id: int, url: str, payload: dict) -> dict:
    """发送单个请求并返回详细结果"""
    start_time = time.time()
    try:
        response = requests.post(
            url,
            headers=HEADERS,
            json=payload,
            timeout=300,  # 增加超时时间
        )
        response.raise_for_status()  # 如果状态码不是 2xx，则引发 HTTPError

        response_time = time.time() - start_time
        result = {
            "request_id": request_id,
            "timestamp": time.time(),
            "status_code": response.status_code,
            "response_time": response_time,
            "success": True,
            "body": response.json(),
            "error": None,
        }
        logging.info(f"Request #{request_id} succeeded in {response_time:.2f}s")

    except requests.exceptions.RequestException as e:
        response_time = time.time() - start_time
        logging.error(f"Request #{request_id} failed after {response_time:.2f}s: {e}")
        result = {
            "request_id": request_id,
            "timestamp": time.time(),
            "status_code": e.response.status_code if e.response else None,
            "response_time": response_time,
            "success": False,
            "body": None,
            "error": str(e),
        }

    return result

def main(args):
    """执行并发测试并输出结果"""
    logging.info(f"开始并发测试: {args.url}")
    logging.info(f"配置: 并发数={args.concurrency}, 总请求数={args.total_requests}")

    start_time = time.time()

    # 加载请求体
    try:
        with open(args.payload_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"无法加载请求体文件: {e}")
        return

    # 使用线程池执行并发请求
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(send_request, i, args.url, payload)
            for i in range(1, args.total_requests + 1)
        ]
        results = [future.result() for future in tqdm(futures, total=args.total_requests)]

    total_time = time.time() - start_time

    # 统计结果
    success_count = sum(1 for r in results if r["success"])
    failed_count = args.total_requests - success_count
    response_times = [r["response_time"] for r in results if r["success"]]
    avg_time = sum(response_times) / success_count if success_count > 0 else 0
    max_time = max(response_times) if response_times else 0
    min_time = min(response_times) if response_times else 0

    # 输出统计摘要
    summary = {
        "total_time_s": round(total_time, 2),
        "total_requests": args.total_requests,
        "success_requests": success_count,
        "failed_requests": failed_count,
        "avg_response_time_s": round(avg_time, 2),
        "max_response_time_s": round(max_time, 2),
        "min_response_time_s": round(min_time, 2),
    }
    logging.info(f"测试完成: {json.dumps(summary, indent=2)}")

    # 保存详细结果到JSON文件
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info(f"详细结果已保存到: {args.output_file}")

    # 输出失败请求详情
    if failed_count > 0:
        logging.warning("检测到失败的请求:")
        for r in results:
            if not r["success"]:
                logging.warning(f"  - 请求 #{r['request_id']}: {r['error']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="并发请求测试工具")
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default="http://127.0.0.1:30300/v1/chat/completions",
        help="目标 URL",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=10, help="并发数"
    )
    parser.add_argument(
        "-n", "--total-requests", type=int, default=100, help="总请求数"
    )
    parser.add_argument(
        "-p",
        "--payload-file",
        type=str,
        required=True,
        help="包含请求体的 JSON 文件",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default=f"request_results_{time.strftime('%Y%m%d_%H%M%S')}.json",
        help="保存详细结果的 JSON 文件",
    )
    args = parser.parse_args()
    main(args)