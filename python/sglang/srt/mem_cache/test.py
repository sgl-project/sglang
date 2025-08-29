import asyncio
import torch
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Any, List, Tuple
import threading
import queue
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryRequest:
    """查询请求的数据结构"""
    query_id: str
    query_tensor: torch.Tensor  # GPU上的query向量
    timestamp: float
    callback: Optional[Callable] = None  # 结果回调函数
    top_k: int = 10
    similarity_threshold: float = 0.0

@dataclass 
class SearchResult:
    """查询结果的数据结构"""
    query_id: str
    indices: torch.Tensor  # 最相似向量的索引
    similarities: torch.Tensor  # 相似度分数
    processing_time: float
    timestamp: float

class SyncAsyncVectorSearcher:
    """支持同步调用的异步向量查找模块"""
    
    def __init__(
        self,
        database_vectors: torch.Tensor,  # 数据库向量 (N, D)
        device: str = "cuda",
        max_concurrent_searches: int = 4,
        batch_size: int = 16,
        similarity_metric: str = "cosine"  # cosine, dot, l2
    ):
        """
        初始化异步向量查找器
        
        Args:
            database_vectors: 数据库中的向量 (N, D)，将被加载到GPU
            device: 计算设备
            max_concurrent_searches: 最大并发搜索数
            batch_size: 批处理大小，用于批量处理查询
            similarity_metric: 相似度度量方法
        """
        self.device = device
        self.batch_size = batch_size
        self.similarity_metric = similarity_metric
        self.max_concurrent_searches = max_concurrent_searches
        
        # 将数据库向量移到GPU并规范化（如果使用cosine相似度）
        self.database_vectors = database_vectors.to(device)
        if similarity_metric == "cosine":
            self.database_vectors = F.normalize(self.database_vectors, p=2, dim=1)
        
        # 同步接口的队列
        self.query_queue_sync = queue.Queue()
        self.result_queue_sync = queue.Queue()
        self.pending_queries = {}  # 存储待处理的查询ID和对应的结果Queue
        
        # 异步组件
        self.loop = None
        self.loop_thread = None
        self.running = False
        
        # 线程池用于GPU计算
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_searches)
        
        logger.info(f"初始化向量查找器: {self.database_vectors.shape[0]} 个向量, 维度: {self.database_vectors.shape[1]}")

    def start(self):
        """启动异步查找服务（同步调用）"""
        if self.running:
            return
        
        self.running = True
        
        # 在新线程中启动事件循环
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
        
        # 等待事件循环启动完成
        time.sleep(0.1)
        
        logger.info("异步向量查找服务已启动")

    def stop(self):
        """停止异步查找服务（同步调用）"""
        if not self.running:
            return
        
        self.running = False
        
        if self.loop and self.loop.is_running():
            # 在事件循环中安排停止任务
            asyncio.run_coroutine_threadsafe(self._async_stop(), self.loop)
        
        # 等待线程结束
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=5.0)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("异步向量查找服务已停止")

    def search(
        self,
        query_tensor: torch.Tensor,
        query_id: Optional[str] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        callback: Optional[Callable[[SearchResult], None]] = None,
        timeout: Optional[float] = None
    ) -> str:
        """
        同步提交查询请求（立即返回query_id，不阻塞）
        
        Args:
            query_tensor: 查询向量 (D,) 或 (1, D)，应该在GPU上
            query_id: 查询ID，如果为None则自动生成
            top_k: 返回top-k相似的向量
            similarity_threshold: 相似度阈值
            callback: 结果回调函数
            timeout: 超时时间（秒）
        
        Returns:
            query_id: 查询ID
        """
        if not self.running:
            raise RuntimeError("搜索服务未启动，请先调用start()")
        
        if query_id is None:
            query_id = f"query_{int(time.time() * 1000000)}"
        
        # 确保query在正确的设备上并且形状正确
        if query_tensor.device != torch.device(self.device):
            query_tensor = query_tensor.to(self.device)
        
        if query_tensor.dim() == 1:
            query_tensor = query_tensor.unsqueeze(0)  # (1, D)
        
        request = QueryRequest(
            query_id=query_id,
            query_tensor=query_tensor,
            timestamp=time.time(),
            callback=callback,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # 将查询放入同步队列
        self.query_queue_sync.put(request)
        
        logger.debug(f"提交查询请求: {query_id}")
        return query_id

    def get_result(self, timeout: Optional[float] = None) -> Optional[SearchResult]:
        """
        获取一个搜索结果（同步调用）
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
        
        Returns:
            result: 搜索结果，如果超时则返回None
        """
        try:
            return self.result_queue_sync.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_all_results(self, timeout: float = 0.1) -> List[SearchResult]:
        """
        获取所有可用的搜索结果（同步调用）
        
        Args:
            timeout: 单次获取的超时时间
        
        Returns:
            results: 搜索结果列表
        """
        results = []
        while True:
            result = self.get_result(timeout=timeout)
            if result is None:
                break
            results.append(result)
        return results

    def _run_event_loop(self):
        """在单独线程中运行事件循环"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._async_main())
        except Exception as e:
            logger.error(f"事件循环发生错误: {e}")
        finally:
            self.loop.close()

    async def _async_main(self):
        """异步主函数"""
        # 启动工作器任务
        tasks = []
        
        # 启动查询传输器
        query_transfer_task = asyncio.create_task(self._query_transfer_worker())
        tasks.append(query_transfer_task)
        
        # 启动搜索工作器
        for i in range(self.max_concurrent_searches):
            worker_task = asyncio.create_task(self._search_worker(f"worker-{i}"))
            tasks.append(worker_task)
        
        # 启动结果传输器
        result_transfer_task = asyncio.create_task(self._result_transfer_worker())
        tasks.append(result_transfer_task)
        
        try:
            # 等待所有任务完成
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            # 取消所有任务
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _async_stop(self):
        """异步停止函数"""
        self.running = False
        # 获取当前循环中的所有任务并取消它们
        tasks = [task for task in asyncio.all_tasks(self.loop) if not task.done()]
        for task in tasks:
            task.cancel()

    async def _query_transfer_worker(self):
        """查询传输工作器：从同步队列传输到异步处理"""
        query_queue_async = asyncio.Queue()
        
        async def queue_monitor():
            while self.running:
                try:
                    # 检查同步队列
                    try:
                        request = self.query_queue_sync.get_nowait()
                        await query_queue_async.put(request)
                    except queue.Empty:
                        await asyncio.sleep(0.001)  # 短暂休眠
                except Exception as e:
                    logger.error(f"查询传输错误: {e}")
        
        # 启动队列监控
        monitor_task = asyncio.create_task(queue_monitor())
        
        # 将异步队列传递给搜索工作器
        self._async_query_queue = query_queue_async
        
        try:
            await monitor_task
        except asyncio.CancelledError:
            monitor_task.cancel()

    async def _search_worker(self, worker_name: str):
        """搜索工作器协程"""
        logger.info(f"搜索工作器 {worker_name} 已启动")
        
        while self.running:
            try:
                # 等待异步查询队列初始化
                if not hasattr(self, '_async_query_queue'):
                    await asyncio.sleep(0.1)
                    continue
                
                # 收集批量请求
                requests = []
                
                # 等待第一个请求
                try:
                    first_request = await asyncio.wait_for(
                        self._async_query_queue.get(), timeout=1.0
                    )
                    requests.append(first_request)
                except asyncio.TimeoutError:
                    continue
                
                # 尝试收集更多请求以形成批次
                start_time = time.time()
                while (len(requests) < self.batch_size and 
                       time.time() - start_time < 0.01):  # 10ms批次收集窗口
                    try:
                        request = await asyncio.wait_for(
                            self._async_query_queue.get(), timeout=0.001
                        )
                        requests.append(request)
                    except asyncio.TimeoutError:
                        break
                
                if requests:
                    # 在线程池中执行GPU计算
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        self.executor, self._search_batch, requests
                    )
                    
                    # 将结果传输到同步队列
                    for result in results:
                        await self._async_result_queue.put(result)
                    
                    logger.debug(f"{worker_name} 处理了 {len(requests)} 个查询")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{worker_name} 发生错误: {e}")
        
        logger.info(f"搜索工作器 {worker_name} 已停止")

    async def _result_transfer_worker(self):
        """结果传输工作器：从异步处理传输到同步队列"""
        self._async_result_queue = asyncio.Queue()
        
        while self.running:
            try:
                result = await asyncio.wait_for(
                    self._async_result_queue.get(), timeout=1.0
                )
                
                # 如果有回调函数，调用它
                if result.callback:
                    try:
                        result.callback(result)
                    except Exception as e:
                        logger.error(f"回调函数执行失败: {e}")
                
                # 将结果放入同步队列
                self.result_queue_sync.put(result)
                
                logger.debug(f"传输查询结果: {result.query_id}")
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"结果传输器发生错误: {e}")

    def _compute_similarity(self, query_batch: torch.Tensor) -> torch.Tensor:
        """
        计算相似度（在GPU上）
        
        Args:
            query_batch: 查询批次 (B, D)
        
        Returns:
            similarities: 相似度矩阵 (B, N)
        """
        if self.similarity_metric == "cosine":
            # query_batch已经在搜索时规范化
            similarities = torch.mm(query_batch, self.database_vectors.t())
        elif self.similarity_metric == "dot":
            similarities = torch.mm(query_batch, self.database_vectors.t())
        elif self.similarity_metric == "l2":
            # L2距离，转换为相似度（距离越小，相似度越高）
            distances = torch.cdist(query_batch, self.database_vectors, p=2)
            similarities = -distances  # 负距离作为相似度
        else:
            raise ValueError(f"不支持的相似度度量: {self.similarity_metric}")
        
        return similarities

    def _search_batch(self, requests: List[QueryRequest]) -> List[SearchResult]:
        """
        批量搜索（GPU计算）
        
        Args:
            requests: 查询请求列表
        
        Returns:
            results: 搜索结果列表
        """
        start_time = time.time()
        
        # 准备批量查询
        query_tensors = [req.query_tensor for req in requests]
        query_batch = torch.cat(query_tensors, dim=0)  # (B, D)
        
        # 规范化查询向量（如果使用cosine相似度）
        if self.similarity_metric == "cosine":
            query_batch = F.normalize(query_batch, p=2, dim=1)
        
        # 计算相似度
        with torch.no_grad():
            similarities = self._compute_similarity(query_batch)  # (B, N)
        
        # 为每个查询生成结果
        results = []
        for i, request in enumerate(requests):
            query_similarities = similarities[i]  # (N,)
            
            # 应用阈值过滤
            if request.similarity_threshold > 0:
                mask = query_similarities >= request.similarity_threshold
                filtered_similarities = query_similarities[mask]
                filtered_indices = torch.nonzero(mask, as_tuple=True)[0]
                
                if len(filtered_similarities) == 0:
                    # 没有满足阈值的结果
                    top_similarities = torch.tensor([], device=self.device)
                    top_indices = torch.tensor([], device=self.device, dtype=torch.long)
                else:
                    # 从过滤后的结果中选择top-k
                    k = min(request.top_k, len(filtered_similarities))
                    top_values, top_local_indices = torch.topk(filtered_similarities, k)
                    top_similarities = top_values
                    top_indices = filtered_indices[top_local_indices]
            else:
                # 直接选择top-k
                k = min(request.top_k, similarities.shape[1])
                top_similarities, top_indices = torch.topk(query_similarities, k)
            
            result = SearchResult(
                query_id=request.query_id,
                indices=top_indices,
                similarities=top_similarities,
                processing_time=time.time() - start_time,
                timestamp=time.time()
            )
            results.append(result)
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "database_size": self.database_vectors.shape[0],
            "vector_dim": self.database_vectors.shape[1],
            "device": self.device,
            "similarity_metric": self.similarity_metric,
            "query_queue_size": self.query_queue_sync.qsize(),
            "result_queue_size": self.result_queue_sync.qsize(),
            "running": self.running
        }

def example_usage():
    """同步主程序使用示例"""
    
    # 创建模拟数据库向量
    database_size = 10000
    vector_dim = 768
    device = "cuda" if torch.cuda.is_available() else "cpu"
    database_vectors = torch.randn(database_size, vector_dim, device=device)
    
    # 创建向量搜索器
    searcher = SyncAsyncVectorSearcher(
        database_vectors=database_vectors,
        device=device,
        max_concurrent_searches=2,
        batch_size=8,
        similarity_metric="cosine"
    )
    
    # 结果回调函数
    def result_callback(result: SearchResult):
        print(f"  └─ 查询 {result.query_id} 完成: "
              f"处理时间={result.processing_time:.4f}s, "
              f"找到{len(result.indices)}个结果, "
              f"最高相似度={result.similarities[0].item():.4f}")
    
    # 启动搜索服务
    searcher.start()
    
    try:
        print("开始同步主循环...")
        
        # 同步主循环
        for step in range(20):
            # 生成新的查询向量
            query = torch.randn(vector_dim, device=device)
            
            # 提交查询（不阻塞）
            query_id = searcher.search(
                query_tensor=query,
                query_id=f"step_{step}",
                top_k=5,
                callback=result_callback
            )
            
            print(f"Step {step}: 提交查询 {query_id}")
            
            # 主循环的其他工作
            time.sleep(0.2)  # 模拟主循环的计算
            
            # 可选：批量获取结果（非阻塞）
            if step % 5 == 0:
                results = searcher.get_all_results(timeout=0.01)
                if results:
                    print(f"  批量获取到 {len(results)} 个结果")
            
            # 检查统计信息
            if step % 10 == 0:
                stats = searcher.get_stats()
                print(f"  统计: 队列={stats['query_queue_size']}, "
                      f"结果队列={stats['result_queue_size']}")
        
        print("\n等待剩余结果...")
        # 主循环结束后，等待所有结果
        remaining_results = []
        while True:
            result = searcher.get_result(timeout=1.0)
            if result is None:
                break
            remaining_results.append(result)
        
        print(f"获取到剩余 {len(remaining_results)} 个结果")
    
    finally:
        # 停止搜索服务
        searcher.stop()
        print("搜索服务已停止")

# 如果直接运行此文件
if __name__ == "__main__":
    example_usage()