import unittest
import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/znorz/下载/sglang/python')

# 直接导入 field_validators 模块
import importlib.util

spec = importlib.util.spec_from_file_location(
    "field_validators",
    "/home/znorz/下载/sglang/python/sglang/srt/utils/field_validators.py"
)
field_validators = importlib.util.module_from_spec(spec)
spec.loader.exec_module(field_validators)

# 从模块中获取函数
validate_list_i64_1d = field_validators.validate_list_i64_1d
validate_optional_list_i64_1d_2d = field_validators.validate_optional_list_i64_1d_2d

class TestFieldValidators(unittest.TestCase):
    
    # ========== validate_list_i64_1d 测试 ==========
    
    def test_validate_list_i64_1d_valid(self):
        """测试有效的 int64 列表"""
        result = validate_list_i64_1d([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])
        
        result = validate_list_i64_1d([0, -1, 100])
        self.assertEqual(result, [0, -1, 100])
        
        result = validate_list_i64_1d([])
        self.assertEqual(result, [])
    
    def test_validate_list_i64_1d_invalid_none(self):
        """测试 None 输入"""
        with self.assertRaises(ValueError) as context:
            validate_list_i64_1d(None)
        self.assertIn("must not be None", str(context.exception))
    
    def test_validate_list_i64_1d_invalid_type(self):
        """测试非列表输入"""
        with self.assertRaises(ValueError) as context:
            validate_list_i64_1d("not a list")
        self.assertIn("must be list", str(context.exception))
        
        with self.assertRaises(ValueError):
            validate_list_i64_1d(123)
    
    def test_validate_list_i64_1d_invalid_element_type(self):
        """测试包含非 int 元素的列表"""
        with self.assertRaises(ValueError) as context:
            validate_list_i64_1d([1, "2", 3])
        # 实际错误消息来自 array('q', v) 的异常
        self.assertIn("contains non-int64 element", str(context.exception))
    
    def test_validate_list_i64_1d_overflow(self):
        """测试超出 int64 范围的数值"""
        huge = 2**63
        with self.assertRaises(ValueError) as context:
            validate_list_i64_1d([1, huge])
        self.assertIn("contains non-int64 element", str(context.exception))
    
    # ========== validate_optional_list_i64_1d_2d 测试 ==========
    
    def test_validate_optional_list_i64_1d_2d_none(self):
        """测试 None 输入（允许的）"""
        result = validate_optional_list_i64_1d_2d(None)
        self.assertIsNone(result)
    
    def test_validate_optional_list_i64_1d_2d_empty(self):
        """测试空列表"""
        result = validate_optional_list_i64_1d_2d([])
        self.assertEqual(result, [])
    
    def test_validate_optional_list_i64_1d_2d_1d(self):
        """测试一维列表"""
        result = validate_optional_list_i64_1d_2d([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])
        
        result = validate_optional_list_i64_1d_2d([10, 20])
        self.assertEqual(result, [10, 20])
    
    def test_validate_optional_list_i64_1d_2d_2d(self):
        """测试二维列表"""
        result = validate_optional_list_i64_1d_2d([[1, 2], [3, 4]])
        self.assertEqual(result, [[1, 2], [3, 4]])
        
        result = validate_optional_list_i64_1d_2d([[10], [20, 30]])
        self.assertEqual(result, [[10], [20, 30]])
        
        result = validate_optional_list_i64_1d_2d([[], []])
        self.assertEqual(result, [[], []])
    
    def test_validate_optional_list_i64_1d_2d_invalid_type(self):
        """测试无效类型输入"""
        with self.assertRaises(ValueError) as context:
            validate_optional_list_i64_1d_2d("not a list")
        self.assertIn("must be list or null", str(context.exception))
        
        with self.assertRaises(ValueError):
            validate_optional_list_i64_1d_2d(123)
    
    def test_validate_optional_list_i64_1d_2d_invalid_elements(self):
        """测试包含无效元素的列表"""
        with self.assertRaises(ValueError) as context:
            validate_optional_list_i64_1d_2d([1, "2", 3])
        # 实际错误消息来自 array('q', v) 的异常
        self.assertIn("contains non-int64 element", str(context.exception))
    
    def test_validate_optional_list_i64_1d_2d_invalid_row(self):
        """测试二维列表中某行无效"""
        with self.assertRaises(ValueError) as context:
            validate_optional_list_i64_1d_2d([[1, 2], [3, "4"]])
        self.assertIn("row 1:", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            validate_optional_list_i64_1d_2d([[1, 2], "invalid"])
        # 实际错误消息来自 validate_list_i64_1d 的检查
        self.assertIn("must be list", str(context.exception))

if __name__ == '__main__':
    unittest.main()
