# LaTeX Resume Template

这是一个专业的 LaTeX 简历模板，适用于 AI Infrastructure / Machine Learning Engineer 岗位。

## 文件说明

- `resume.tex`: 主简历文件（LaTeX 源代码）
- `README.md`: 使用说明

## 使用方法

### 在 Overleaf 中使用

1. **创建新项目**：
   - 访问 [Overleaf](https://www.overleaf.com/)
   - 点击 "New Project" → "Upload Project"
   - 上传 `resume.tex` 文件

2. **编译**：
   - Overleaf 会自动检测 LaTeX 文件
   - 点击 "Recompile" 按钮生成 PDF
   - 推荐使用 **XeLaTeX** 或 **pdfLaTeX** 编译器

3. **编辑内容**：
   - 修改个人信息（姓名、联系方式等）
   - 更新教育背景、工作经验、项目经历
   - 调整技能列表和荣誉奖项

### 本地编译

如果你在本地安装了 LaTeX 环境（如 TeX Live 或 MiKTeX）：

```bash
# 使用 pdfLaTeX
pdflatex resume.tex

# 或使用 XeLaTeX（支持更好的字体和 Unicode）
xelatex resume.tex
```

## 自定义选项

### 1. 更改主题风格

```latex
\moderncvstyle{classic}  % 选项: 'casual', 'classic', 'oldstyle', 'banking'
```

### 2. 更改颜色主题

```latex
\moderncvcolor{blue}  % 选项: 'blue', 'orange', 'green', 'red', 'purple', 'grey', 'black'
```

### 3. 调整页面边距

```latex
\usepackage[scale=0.75]{geometry}  % 调整 scale 值（0.5-1.0）
```

### 4. 添加照片

```latex
\photo[64pt][0.4pt]{picture}  % 将 'picture' 替换为你的照片文件名
```

## 模板特点

✅ **现代化设计**：使用 `moderncv` 包，专业美观  
✅ **Overleaf 兼容**：可直接在 Overleaf 上编译  
✅ **适合技术岗位**：针对 AI Infra / ML Engineer 优化  
✅ **完整结构**：包含所有常见简历章节  
✅ **易于定制**：清晰的代码结构，易于修改  

## 包含的章节

- **Summary**：个人简介/总结
- **Education**：教育背景
- **Experience**：工作经验
- **Projects**：项目经历
- **Technical Skills**：技术技能
- **Publications**：论文/出版物（可选）
- **Certifications**：认证证书（可选）
- **Awards & Honors**：奖项荣誉（可选）
- **Additional Information**：附加信息（可选）

## 注意事项

1. **个人信息**：记得修改所有的占位符内容
2. **照片**：如果需要添加照片，请准备一张正方形照片（推荐 64pt 尺寸）
3. **链接**：GitHub、LinkedIn 等链接会自动格式化为可点击的链接
4. **中文支持**：如果需要添加中文内容，建议使用 XeLaTeX 编译器并配置中文字体

## 常见问题

### Q: 如何在 Overleaf 中添加照片？
A: 在 Overleaf 项目中上传照片文件，然后在 `\photo` 命令中使用文件名。

### Q: 如何调整简历长度？
A: 修改 `\usepackage[scale=0.75]{geometry}` 中的 `scale` 值。较小的值（如 0.65）会让内容更紧凑。

### Q: 如何隐藏某个章节？
A: 注释掉或删除对应的 `\section{}` 和 `\cvitem{}` 或 `\cventry{}` 内容。

### Q: 编译错误怎么办？
A: 
- 确保使用正确的编译器（XeLaTeX 或 pdfLaTeX）
- 检查所有大括号是否匹配
- 查看 Overleaf 的编译日志找出错误位置

## 推荐资源

- [Overleaf 官方文档](https://www.overleaf.com/learn)
- [moderncv 包文档](https://www.ctan.org/pkg/moderncv)
- [LaTeX 入门指南](https://www.overleaf.com/learn/latex/Learn_LaTeX_in_30_minutes)

## 许可证

此模板可自由使用和修改。
