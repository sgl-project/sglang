# JD Processing Workflow

## 使用流程

1. **输入 JD**：粘贴完整的 Job Description
2. **系统判断**：
   - 检查硬拒绝条件（签证、执照、10+ years）
   - 选择模版（LLM/ML 或 Hardware/Imaging）
   - 提取 JD 关键词
3. **生成 LaTeX**：
   - 只输出 3 段 Experience（HydroSense / Seno / Roswell）
   - 格式：544 bullets（HydroSense: 5, Seno: 4, Roswell: 4）
   - 每条 ~28 words

## 输出格式

### 如果触发硬拒绝条件：
```
不合适，下一个
```

### 如果通过筛选：
输出完整的 LaTeX 代码（只包含 Experience 部分）：
```latex
\cventry{2021--Present}{Co-founder / [Title from JD]}{HydroSense}{City, State}{}{
  \begin{itemize}
    \item Bullet 1 (~28 words)
    \item Bullet 2 (~28 words)
    \item Bullet 3 (~28 words)
    \item Bullet 4 (~28 words)
    \item Inventor on 3 granted patents related to [relevant field].
  \end{itemize}}

\cventry{2019--2021}{[Title from JD]}{Seno}{City, State}{}{
  \begin{itemize}
    \item Bullet 1 (~28 words)
    \item Bullet 2 (~28 words)
    \item Bullet 3 (~28 words)
    \item Bullet 4 (~28 words)
  \end{itemize}}

\cventry{2017--2019}{[Title from JD]}{Roswell}{City, State}{}{
  \begin{itemize}
    \item Bullet 1 (~28 words)
    \item Bullet 2 (~28 words)
    \item Bullet 3 (~28 words)
    \item Bullet 4 (~28 words)
  \end{itemize}}
```

## 注意事项

- **不输出解释文字**：只输出 LaTeX 代码
- **不分析匹配度**：只执行改写
- **不劝申不申**：除非触发硬拒绝条件
- **严格遵循规则**：544 结构、~28 words、HydroSense 第 5 条必须是专利
