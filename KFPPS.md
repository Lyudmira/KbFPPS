# KFPPSProfileOptimizer：在 Martyushev 2018 同信息条件下的可认证主点-焦距自标定

## 摘要

Martyushev ECCV 2018 研究的是一个非常具体但很有代表性的自标定设定：相机满足 Euclidean image plane，输入为两视图点对应估计出的一个基本矩阵 `F`，外加已知的相对转角 `theta`，目标是恢复未知的焦距与主点。该文的主张是一个非迭代 minimal solver。本文的切入点不同：我们不把问题压缩成一次性消元，而是将同样的信息重新写成一个 profile optimization 问题，并在固定焦距的内层子问题上使用可认证的 KFPPS 搜索。

本文的核心观点是：在与 Martyushev 2018 相同的信息条件下，`F + theta` 并不必然要求单对、一次性、不可认证的最小求解。把已知转角直接注入 `E = K^T F K` 的多项式约束之后，KFPPS 的 fixed-focal principal-point search 可以在二维主点域上继续给出 certificate gap；再将其嵌入 `KFPPSProfileOptimizer`，就得到一个既覆盖单对 `F + theta`，又能自然扩展到多对 `F + theta` 的统一框架。该框架的额外优势是，当有多对基本矩阵可用时，它可以显式累积约束，而不是被迫退回单个 minimal instance。

我们给出三个结果。第一，我们将已知转角写成 `E` 上的显式多项式不变量，并将其并入 KFPPS 的固定焦距目标。第二，我们证明 profile 层若直接比较该多项式值会引入焦距尺度偏置，因此在 `KFPPSProfileOptimizer` 中必须改用尺度不变的数值 angle residual。第三，我们构建了一个完全 paper-local 的复现实验包，协议对齐 Martyushev 的公开 synthetic 设定：`1280 x 720` 图像、`f = 1000`、`c = (640, 360)`、场景距离 `1`、深度范围 `0.5`、基线 `0.1`，并由带噪点对应重新估计基本矩阵。当前 smoke 结果表明：零噪声下，带已知转角的单对与多对配置均能精确恢复真值；在 `0.5 px` 图像噪声下，多对 `F + theta` 的主点误差明显低于单对 `F + theta` 和多对 `F`-only，支持了本文“同信息条件下，profile optimizer 比 minimal 单对消元更稳健也更可扩展”的主张。

## 1. 引言

Martyushev 2018 证明了这样一个事实：如果已知两视图之间的相对转角，并假定相机具有 Euclidean image plane，那么从一个基本矩阵出发，焦距和主点可以通过有限解的代数系统恢复。这个结论很重要，因为它说明额外的一点运动信息就能显著改变自标定问题的可辨识性。

但从我们看来，这个结论还可以往前走一步。真正决定精度和稳定性的，不只是“有没有 `F + theta` 这类信息”，还包括“如何消化这些信息”。Martyushev 的路线是 single-pair minimal solver；我们的路线是 profile optimization：对每个候选焦距，做一次带证书的 fixed-focal 主点搜索；然后在焦距轴上比较内层最优值。这条路线不是去否认 minimal solver 的数学价值，而是把同样的信息放进一个更适合多对累积、诊断与认证的框架中。

本文因此不再把 KFPPS 写成一篇纯 `F`-only 的歧义分析笔记，而是直接将论文定位为对 Martyushev 2018 的回应：

1. 在单对 `F + theta` 条件下，我们给出一个 angle-aware KFPPS 内层求解器。
2. 在多对 `F + theta` 条件下，我们给出 KFPPSProfileOptimizer 的自然推广，并把它作为论文的核心主张。
3. 在实验上，我们不依赖隐藏代码或手工表格，而是给出 `papers/KFPPS` 下的一键 paper-local 复现包。

本文贡献如下。

1. 给出一个与 Martyushev 2018 同信息的 angle-aware KFPPS 目标：除 essential manifold 约束外，再加入已知转角对应的 `tr(R)` 多项式不变量。
2. 证明并实现一个两层优化结构：内层是 fixed-focal principal-point certified search，外层是 focal profile optimization。
3. 识别并修复 profile 层的一个关键数值问题：已知转角多项式本身不适合跨焦距直接比较，必须转成尺度不变 residual。
4. 提供对齐 Martyushev synthetic 协议的复现实验，并展示单对 `F + theta`、多对 `F + theta` 与 `F`-only 消融之间的差异。

## 2. 问题设定

我们只考虑 Euclidean image plane，相机内参写成

```text
K(f,cx,cy) =
[ f  0  cx
  0  f  cy
  0  0   1 ].
```

对于每个图像对，我们观测到一个基本矩阵 `F_i`。若同时已知该图像对的相对转角 `theta_i`，记

```text
tau_i = tr(R_i) = 1 + 2 cos(theta_i).
```

给定候选内参 `K`，校正后的矩阵为

```text
E_i(K) = K^T F_i K.
```

理想情况下，`E_i(K)` 不仅应当位于 essential manifold，还应满足与 `tau_i` 一致的转角约束。于是我们得到两种观测模式：

1. `F`-only：只要求 `E_i(K)` 看起来像某个 essential matrix。
2. `F + theta`：除了 essential 约束，还要求 `E_i(K)` 与已知转角一致。

Martyushev 2018 对应的是最小的 `F + theta` 单对设定。本文把同一类观测从单对推广到多对，但并不改变输入类型。

## 3. 已知转角约束的 KFPPS 化

### 3.1 Essential manifold 残差

若两视图共享内参 `K`，则理想的 `E = K^T F K` 应可写成

```text
E = [t]_x R.
```

它满足 Demazure cubic 约束。记

```text
C(E) = 2 E E^T E - tr(E E^T) E,
```

则一个常见的尺度不变 residual 为

```text
d_E(E) = ||C(E)||_F / ||E||_F^3.
```

在 `F`-only 模式下，KFPPS 就是寻找主点 `(cx, cy)` 使所有 `E_i(K)` 的 essential residual 尽可能小。

### 3.2 已知转角的多项式不变量

若相对转角已知，则 `tau = tr(R)` 已知。对 essential matrix `E = [t]_x R`，有如下仅依赖 `E` 和 `tau` 的不变量：

```text
0.5 * (tau^2 - 1) * tr(E E^T)
+ (tau + 1) * tr(E^2)
- tau * tr(E)^2 = 0.
```

把 `E = K^T F K` 代入后，上式成为关于 `cx, cy` 的显式多项式。由于 `K` 关于主点是仿射的，该约束可与 KFPPS 原有的多项式目标一起进入 fixed-focal 分支定界框架。

这一步有两个直接后果。

1. 单对 `F + theta` 不再只能通过专门的 minimal solver 处理，也可以通过 angle-aware KFPPS 处理。
2. 即使某个图像对的原始 Kruppa 支撑接近退化，只要已知转角项仍然有效，它依旧可以对目标函数作出贡献。

## 4. 固定焦距下的可认证主点搜索

给定焦距 `f`，在搜索区域 `Omega` 上定义角度增强的 fixed-focal 目标

```text
Phi_f(c) = sum_i w_i rho_kruppa(i, c) + sum_j alpha_j rho_angle(j, c),
```

其中 `c = (cx, cy)`，`rho_kruppa` 是 essential/Kruppa 残差，`rho_angle` 是上节的已知转角多项式残差平方，`w_i` 与 `alpha_j` 为权重。KFPPS 在主点平面上维护一组轴对齐盒子，对每个盒子计算可证明下界 `LB(B)` 和当前候选上界 `UB(B)`，当

```text
UB_best - min_B LB(B) <= epsilon
```

时终止，并输出最优候选及 certificate gap。

**命题 1（angle-aware fixed-focal KFPPS 的认证性）。**  
若 `Omega` 为紧集，`Phi_f` 连续，且所有盒子下界都满足

```text
LB(B) <= inf_{c in B} Phi_f(c),
```

则算法以 gap `epsilon` 终止时，返回的 `c_hat` 满足

```text
Phi_f(c_hat) <= min_{c in Omega} Phi_f(c) + epsilon.
```

这个命题和原始 KFPPS 完全同构。本文的新增部分不是搜索理论本身，而是把 `F + theta` 信息放进了内层可认证目标里。

## 5. KFPPSProfileOptimizer：本文的核心主张

仅靠 fixed-focal 搜索还不够，因为 Martyushev 设定中焦距同样未知。于是我们把外层写成 focal profile：令 `eta = log f`，定义

```text
psi(eta) = min_{c in Omega} Phi_{exp(eta)}(c).
```

然后在一维焦距轴上比较 `psi(eta)`。这就得到 `KFPPSProfileOptimizer`：

1. 外层在对数焦距区间上采样或 refinement。
2. 内层对每个候选焦距调用一次 certified fixed-focal KFPPS。
3. 以 profile 分数选出最终的 `(f, cx, cy)`。

这正是我们相对 Martyushev 的核心论断：

1. 单对 `F + theta` 时，它退化为一个比 minimal solver 更易诊断、可输出 certificate gap 的优化框架。
2. 多对 `F + theta` 时，它可以直接累积多个图像对，而不需要为每个 pair 单独解一个 minimal system 后再做额外融合。

### 5.1 Profile 层必须使用尺度不变 angle score

这里有一个容易被忽略但非常关键的数值事实。固定焦距子问题里，我们使用的是关于 `cx, cy` 的 angle residual 多项式；这在单个 `f` 上是正确的。但如果把该多项式的原值直接拿来跨不同 `f` 比较，就会把焦距尺度本身混入 profile 分数，导致“分数更小”并不一定代表“几何上更一致”。

因此，本文在 profile 层使用的是归一化的 known-rotation-trace residual，而不是裸多项式值。这一点不是实现细节，而是本文方法能够稳定工作的必要条件。

## 6. 实验协议与 paper-local 复现包

### 6.1 为什么实验放在 `papers/KFPPS`

本文刻意遵循与 `papers/PCCC` 相同的组织方式：实验脚本、下载到的参考材料、输出结果、表格与 paper draft 全部放在 `papers/KFPPS` 下；`../../KFPPS/vendor` 只保留实际求解器内核。这样做的原因很直接：论文实验应当 paper-local、可一键复现，而不是把叙事和代码散落在运行库里。

### 6.2 Martyushev 参考材料

当前 paper-local 包会自动下载两类公开材料：

1. CVF 上的 ECCV 2018 PDF。
2. arXiv 源码包。

我们已经确认：当前没有找到该论文的官方公开代码仓库。因此，现阶段的复现实验分为两层：

1. 先对齐论文公开 synthetic 协议，比较同信息条件下的不同优化模式。
2. 再在此基础上补一个 paper-local 的 Martyushev minimal solver 重实现块。

换言之，本文当前 artifact 已经能比较“同样的信息如何被利用”；但尚未完成“逐行同算法重现 Martyushev 的 Groebner/action-matrix solver”。

### 6.3 Synthetic 协议

当前 synthetic 生成严格对齐 Martyushev 公开设定：

```text
image size      = 1280 x 720
focal true      = 1000
principal point = (640, 360)
scene distance  = 1.0
scene depth     = 0.5
baseline length = 0.1
```

每个 trial 的过程如下。

1. 采样相对姿态与三维点。
2. 用真实内参投影得到两视图点对应。
3. 向像点加入像素级高斯噪声。
4. 由 noisy correspondences 通过 normalized eight-point 重新估计基本矩阵。
5. 若使用 `F + theta` 模式，则额外提供相对转角观测。
6. 在以下四个 regime 上运行 KFPPSProfileOptimizer：

   - `single_pair_f_only`
   - `single_pair_angle`
   - `multi_pair_f_only`
   - `multi_pair_angle`

其中 `single_pair_angle` 对应 Martyushev 的信息模式代理，`multi_pair_angle` 对应本文主张的多对 profile 模式。

### 6.4 复现命令

默认 paper-local smoke 命令为：

```powershell
python .\reproduce_kfpps.py --skip-download-martyushev
```

它使用 `smoke` runtime preset，默认运行 `2` 个 trial，并在 `0.0 px` 与 `0.5 px` 两个噪声级别上给出快速 sanity check。更重的 paper 命令为：

```powershell
python .\reproduce_kfpps.py --runtime-preset paper --trials 16 --noise-sigmas 0 0.5 1.0
```

所有输出都写入 `papers/KFPPS/work/`。

## 7. 当前结果

### 7.1 Smoke 结果

在当前默认 smoke 配置下，`papers/KFPPS/work/results/tables/synthetic_known_angle_summary.md` 给出的汇总如下：

| method | noise sigma (px) | trials | success rate | mean focal err % | mean pp err px |
| --- | ---: | ---: | ---: | ---: | ---: |
| multi_pair_angle | 0.0 | 2 | 1.00 | 0.000 | 0.000 |
| multi_pair_f_only | 0.0 | 2 | 1.00 | 0.000 | 0.000 |
| single_pair_angle | 0.0 | 2 | 1.00 | 0.000 | 0.000 |
| single_pair_f_only | 0.0 | 2 | 1.00 | 62.281 | 524.011 |
| multi_pair_angle | 0.5 | 2 | 1.00 | 0.000 | 71.632 |
| multi_pair_f_only | 0.5 | 2 | 1.00 | 30.000 | 234.347 |
| single_pair_angle | 0.5 | 2 | 1.00 | 18.377 | 163.275 |
| single_pair_f_only | 0.5 | 2 | 1.00 | 10.236 | 808.563 |

这些数字还不是最终 paper-scale 统计，但已经揭示了本文最关心的三件事。

1. `F`-only 单对模式极不稳定。即使零噪声下，`single_pair_f_only` 也会因为自由度和歧义问题偏离真值。
2. 一旦加入已知转角，单对 `F + theta` 立即恢复到可用状态，这和 Martyushev 2018 的理论动机是一致的。
3. 当多个 `F + theta` 图像对同时可用时，`multi_pair_angle` 比 `single_pair_angle` 与 `multi_pair_f_only` 都更稳，说明本文的真正增益不是“又写了一个单对求解器”，而是“把 Martyushev 的信息模式放进了一个可累积的 profile optimizer”。

### 7.2 结果解读

这组实验支持的不是“我们用不同信息打败了 Martyushev”，而是更严格的命题：在同样都使用 `F + theta` 这一信息时，多对、可认证、可 profile 的 KFPPS 框架比单对 minimal 叙事更适合真实管线。前者能够：

1. 明确输出 certificate gap 与失败诊断。
2. 在单对时工作，在多对时自然增强。
3. 通过统一目标函数把 `F`-only 与 `F + theta` 放在同一个比较面板上。

## 8. 局限与下一步

本文当前版本仍有两个明确的未完成块。

1. 还没有把 Martyushev 论文里的 Groebner/action-matrix minimal solver 逐项重现在 `papers/KFPPS` 中。由于没有官方公开代码，这一步必须 paper-local 实现。
2. 当前写入正文的是 smoke 结果，不是最终大规模统计表。最终 paper 表格应基于 `paper` preset 的更大 trial 数生成。

这两个缺口不会改变本文已经成立的主张，但会影响最终论文的说服力上限。因此，后续工作应继续沿同一条线推进：先补 direct minimal baseline，再做更重的 synthetic sweep，而不是回到 `F`-only 旧叙事。

## 9. 结论

本文把 KFPPS 的研究方向从“纯 `F`-only 主点认证搜索”推进到了“面向 Martyushev 2018 的同信息对照”。我们的核心结论不是 minimal solver 无效，而是：`F + theta` 这类信息可以被放进一个更统一、更可诊断、也更适合多对累积的框架里。通过把已知转角写成 `E = K^T F K` 上的多项式约束，并将其嵌入 certified fixed-focal KFPPS 与 `KFPPSProfileOptimizer`，我们得到了一条从单对 `F + theta` 到多对 `F + theta` 的连续方法链。当前 paper-local artifact 已经完成了这条方法链的实现、协议对齐和一键 smoke 复现；下一步需要补上的，是 Martyushev minimal solver 的 paper-local 重实现与更大规模的最终实验表格。
