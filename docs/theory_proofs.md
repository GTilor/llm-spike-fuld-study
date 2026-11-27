# 先验尺度托底的残差自适应优化器：机制理论与严格证明

本文档给出单遍流式领域切换与局部重尾梯度冲击下，**先验尺度托底残差分数阶优化器 `T13`（`Prior-Floored AdaFULD-W`）**及其同构单调平顶核 `T13_FlatTop` 的精确递推定义与机制分析命题。

---

## 1. 算法精确递推定义

对每个参数张量 $W \in \mathbb{R}^p$ 及步数 $t \ge 1$（默认超参数：尾指数 $\alpha = 1.5$，新息信任倍数 $\tau = 3.0$，二阶矩门控倍数 $\kappa = 2.0$，先验托底比例 $\rho_{\text{prior}} = 0.15$，分母托底 $\rho_{\text{denom}} = 0$，数值保护 $\epsilon = 10^{-8}, \epsilon_0 = 10^{-12}$）：

- **首步初始化（$t = 1$）**：$\tilde{g}_1 = g_1, \; \bar{g}_1 = g_1$。
- **流式迭代（$t \ge 2$）**：记历史偏差校正一阶矩与二阶矩 $M_{t-1} = \frac{m_{t-1}}{1 - \beta_1^{t-1}}$，$V_{t-1} = \frac{v_{t-1}}{1 - \beta_2^{t-1}}$，同张量历史均方根尺度 $\bar{\sigma}_{t-1} = \sqrt{\frac{1}{p}\sum_{j=1}^p V_{t-1, j}}$：
  1. **先验创新尺度托底（Prior-Only Subspace Scale Borrowing）**：
     $$s_{t-1, i} = \max\!\left(\sqrt{V_{t-1, i}}, \; \rho_{\text{prior}} \, \bar{\sigma}_{t-1}\right) + \epsilon, \qquad h_{t, i} = \tau \, s_{t-1, i}, \qquad z_{t, i} = \frac{g_{t, i} - M_{t-1, i}}{h_{t, i}}.$$
  2. **零观测保持与一阶新息核滤波**：
     $$\tilde{g}_{t, i} = \begin{cases} 0, & g_{t, i} = 0, \\ M_{t-1, i} + h_{t, i}\,\phi(z_{t, i}), & g_{t, i} \ne 0, \end{cases}$$
     其中 `T13` 取 $\phi = \psi_\alpha$，即原点单位斜率归一化的对称 $\alpha$-稳定分布负对数密度导数（在 [adafuld/optimizers.py](../adafuld/optimizers.py) 中由严格无实极点的 $(3,4)$ 阶有理函数与 $|z|>60$ 处首项渐近 $c_\alpha / z$ 连续拼接实现；实现核在 $\alpha = 1.5$ 时的精确正峰位置为 $z_* = 1.7185172$，峰高 $\psi_{\max} = 1.0413236$）；峰前逐位同构平顶对照核 `T13_FlatTop` 取 $\phi = P_\alpha(z) := \operatorname{sgn}(z)\psi_\alpha(\min(|z|, z_*))$。
  3. **阈值内零失真二阶矩门控**：
     $$\bar{g}_{t, i} = g_{t, i} \cdot \min\!\left(1, \; \frac{|M_{t-1, i}| + \kappa \, h_{t, i}}{|g_{t, i}| + \epsilon_0}\right).$$
  4. **矩更新与预条件步**：
     $$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\tilde{g}_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\bar{g}_t^2,$$
     $$M_t = \frac{m_t}{1 - \beta_1^t}, \qquad V_t = \frac{v_t}{1 - \beta_2^t}, \qquad \sigma_{t, i} = \max\!\left(\sqrt{V_{t, i}}, \; \rho_{\text{denom}}\bar{\sigma}_t\right) + \epsilon, \qquad \theta_{t, i} = (1 - \eta\lambda)\theta_{t-1, i} - \eta \frac{M_{t, i}}{\sigma_{t, i}}.$$

---

## 2. 机制定理与证明

### 命题 1（标准 AdamW 休眠脉冲放大 vs. 有界核脉冲上界）

设在第 $t_0 \ge 2$ 步前，休眠坐标 $i \in \mathcal{S}$ 满足 $m_{t_0-1, i} = v_{t_0-1, i} = 0$，且 $0 < \beta_1 < \sqrt{\beta_2} < 1$。考虑孤立脉冲：第 $t_0$ 步收到 $g_{t_0, i} = g \ne 0$，随后 $H - 1$ 步满足 $g_{t_0+s, i} = 0$（$1 \le s \le H - 1$），设 $\lambda = 0$。

1. **（标准 AdamW 的 $(1 - \beta_2)^{-1/2}$ 阶放大）**：在标准 `AdamW` 中，单步无学习率更新幅值为
   $$|u_{t_0+s, i}^{\text{AdamW}}| = \frac{(1 - \beta_1)\beta_1^s |g| / (1 - \beta_1^{t_0+s})}{|g|\sqrt{(1 - \beta_2)\beta_2^s / (1 - \beta_2^{t_0+s})} + \epsilon}, \qquad \lim_{H \to \infty}\lim_{t_0 \to \infty}\lim_{\epsilon \to 0^+} \sum_{s=0}^{H-1} |u_{t_0+s, i}^{\text{AdamW}}| = \frac{1 - \beta_1}{\sqrt{1 - \beta_2}(1 - \beta_1/\sqrt{\beta_2})}.$$
   在 $t_0 = 601, \beta_1 = 0.9, \beta_2 = 0.995, \epsilon \to 0^+$ 下，单次孤立脉冲累计位移达 **`14.1248` 个学习率单位**。
2. **（双侧同层下界 $\rho_{\text{prior}} = \rho_{\text{denom}} = \rho > 0$ 的一致上界）**：若同张量活跃尺度满足 $\bar{\sigma}_{t_0+s} \ge \gamma_{\text{act}} \bar{\sigma}_{t_0-1}$（$\forall 0 \le s \le H-1, \; \gamma_{\text{act}} > 0$），则对任意满足 $|\phi(z)| \le \psi_{\max}$ 的有界核（含 $\psi_\alpha$ 与 $P_\alpha$）及任意 $|g| > 0$：
   $$\sum_{s=0}^{H-1} |u_{t_0+s, i}| \le \frac{\tau\psi_{\max}(1 - \beta_1^H)}{1 - \beta_1^{t_0}} \cdot \frac{\rho\bar{\sigma}_{t_0-1} + \epsilon}{\gamma_{\text{act}}\rho\bar{\sigma}_{t_0-1} + \epsilon} \le \frac{\tau\psi_{\max}}{\min(\gamma_{\text{act}}, 1)\,(1 - \beta_1^{t_0})}.$$

*证明*：由 $|\tilde{g}_{t_0, i}| \le \tau\psi_{\max}(\rho\bar{\sigma}_{t_0-1}+\epsilon)$、后续 $s \ge 1$ 步 $\tilde{g}_{t_0+s, i} = 0$，以及分母 $\sigma_{t_0+s, i} \ge \gamma_{\text{act}}\rho\bar{\sigma}_{t_0-1} + \epsilon \ge \min(\gamma_{\text{act}}, 1)(\rho\bar{\sigma}_{t_0-1}+\epsilon)$，对几何级数 $\sum_{s=0}^{H-1}(1-\beta_1)\beta_1^s/(1-\beta_1^{t_0+s}) \le (1-\beta_1^H)/(1-\beta_1^{t_0})$ 求和即得。 $\square$

---

### 命题 2A（二阶矩阈值内零失真与单步防污染界）

令 $C_{t, i} = |M_{t-1, i}| + \kappa h_{t, i}$。
1. 当 $g_{t, i} = 0$ 时 $\bar{g}_{t, i} = 0$；当 $0 < |g_{t, i}| \le C_{t, i} - \epsilon_0$ 时，$\frac{C_{t, i}}{|g_{t, i}| + \epsilon_0} \ge 1$，故 $\bar{g}_{t, i} = g_{t, i}$ **精确成立**（阈值内零失真）。
2. 当稳态坐标（$V_{t-1, i} = \sigma_0^2, |M_{t-1, i}| \le c_m \sigma_0, s_{t-1, i} = \gamma_s \sigma_0$）遭遇任意大离群冲击 $|g_{t, i}| = M \sigma_0$ 时，记 $b_t = \frac{1 - \beta_2}{1 - \beta_2^t}, K = c_m + \kappa\tau\gamma_s$，有 $|\bar{g}_{t, i}| \le K\sigma_0$，从而
   $$\frac{\sqrt{V_{t, i}}}{\sigma_0} \le \sqrt{1 + b_t(K^2 - 1)}.$$
   对 $t=600, \beta_2=0.995, c_m=1, \kappa=2, \tau=3, \gamma_s=1$，上界与冲击倍数 $M$ 无关且不超过 **`1.1191`**（而未门控 `AdamW` / `StableAdamW` 在 $M=300$ 时单步膨胀达 **`21.7804` 倍**）。 $\square$

---

### 命题 2B（无先验托底 `C11` 的唤醒爬坡陷阱 vs. `T13` 的首步捕获定理）

设休眠新域坐标 $i$ 在切换步 $t_0 \gg 1$ 前满足 $M_{t_0-1, i} = V_{t_0-1, i} = 0$，切入新域后收到真实非零梯度序列 $g_{t_0+k, i} = g$，其幅值处于正常同层尺度 $|g| = \gamma_g \bar{\sigma}_{t_0-1} \gg \epsilon$。开启二阶矩门控 `v_filter="cap"`：

1. **（无先验托底 $\rho_{\text{prior}} = 0$ 时的首步反比压制与对数爬坡延迟）**：
   若 $\rho_{\text{prior}} = 0$，初始尺度 $s_{t_0-1, i} = \epsilon$。在第 $t_0$ 步（首次唤醒）：
   - 平顶核（$P_\alpha$）的一阶输入幅值为 $\tau\epsilon\psi_{\max}$（对 $\tau=3, \epsilon=10^{-8}, \alpha=1.5$ 为 $3.124 \times 10^{-8}$）；
   - 回落核 `C11`（$\psi_\alpha$）在 $|z_{t_0, i}| = \frac{|g|}{\tau\epsilon} \gg z_*$ 下，一阶输入被反比例压小至 $|\tilde{g}_{t_0, i}| \le \frac{(c_\alpha/P_0) \tau^2 \epsilon^2}{|g|}$（对 $|g| = 6 \times 10^{-4}$ 仅为 $2.954 \times 10^{-12}$，较平顶核初始额外压制逾 **$10^4$ 倍**）。

   定义门控包络 $D_t := |M_{t, i}| + \kappa\tau(\sqrt{V_{t, i}} + \epsilon)$（初值 $D_{t_0-1} = \kappa\tau\epsilon$）。在解除门控前（$|\bar{g}_t| \le D_{t-1}$），记 $a_t = \frac{1-\beta_1}{1-\beta_1^t}, \; b_t = \frac{1-\beta_2}{1-\beta_2^t}$，由 $|M_t - M_{t-1}| \le a_t|\tilde{g}_t - M_{t-1}| \le a_t \tau\psi_{\max}(\sqrt{V_{t-1}}+\epsilon) \le \frac{a_t\psi_{\max}}{\kappa} D_{t-1}$ 及 $\sqrt{V_t} + \epsilon \le \sqrt{V_{t-1}} + \epsilon + \sqrt{b_t} D_{t-1}$，严格成立线性包络递推：
   $$D_t \le \left(1 + \kappa\tau\sqrt{b_t} + \frac{a_t \psi_{\max}}{\kappa}\right) D_{t-1}.$$
   故解除二阶门控（$D_t \ge |g|$）所需的有效出现次数满足严格解析下界：
   $$K_{\text{trap}} \ge \frac{\ln(|g| / (\kappa\tau\epsilon))}{\ln\!\left(1 + \kappa\tau\sqrt{b_{t_0}} + \frac{a_{t_0}\psi_{\max}}{\kappa}\right)}.$$
   对典型参数 $|g|=6\times 10^{-4}, t_0=601, \beta_1=0.9, \beta_2=0.995, \tau=3, \kappa=2$，解析下界保证至少需 **`24` 次**非零出现；直接执行精确双精度递推，需至**第 `94` 次**出现才首次完整写入 $g^2$，**第 `95` 次**出现才首次进入峰前区 $|z| \le z_*$。

2. **（开启先验托底 `T13` $\rho_{\text{prior}} > 0$ 的首步捕获与四次多项式余项界）**：
   在 `T13` 中，$s_{t_0-1, i} = \rho_{\text{prior}}\bar{\sigma}_{t_0-1} + \epsilon$：
   - **二阶矩首步完整写入**：只要 $|g| + \epsilon_0 \le \kappa\tau(\rho_{\text{prior}}\bar{\sigma}_{t_0-1} + \epsilon)$，二阶矩在**第 1 次出现**即零失真完整写入 $\bar{g}_{t_0, i} = g$（$K_{\text{trap}} = \mathbf{0}$）；
   - **一阶峰内多项式相对收缩双侧界**：对实现核 $\alpha = 1.5$（三次系数 $c_3 = Q_0 - P_1/P_0 = 0.1660210$），在区间 $0 < |z| \le 1$ 上直接展开分子分母有理项得严格界：
     $$0 \le c_3 z^2 - \left(1 - \frac{\psi_{1.5}(z)}{z}\right) \le 0.01615 \, z^4.$$
     例如当 $\gamma_g = |g| / \bar{\sigma}_{t_0-1} = 0.16, \rho_{\text{prior}} = 0.15, \tau = 3$ 时，$|z_{t_0, i}| \approx 0.3556 \le 1$，首步一阶相对收缩仅为 **`2.077%`**（通过率 **`97.923%`**）。 $\square$

---

### 命题 3B（`T13` 的分子先验尺度自相消性质 vs. 大 $\epsilon$ 的低频词阻尼）

考虑低频特征坐标 $w$，其在批次中稀疏出现，真实梯度满足 $|g_{t, w} - M_{t-1, w}| \ll h_{t, w}$ 且历史二阶矩满足 $\sqrt{V_{t, w}} \lesssim \epsilon_{\text{large}} = 10^{-4}$：
1. **（加性大 $\epsilon$ `B18` 或分母下界的步长压缩）**：在 $\sigma_{t, w} = \sqrt{V_{t, w}} + \epsilon_{\text{large}}$ 下，每次更新步长相对标准 `AdamW` 被乘性压缩为 $\frac{\sqrt{V_{t, w}} + 10^{-8}}{\sqrt{V_{t, w}} + \epsilon_{\text{large}}} \ll 1$。
2. **（`T13` 的分子尺度自相消）**：在 `T13`（$\rho_{\text{prior}} > 0, \rho_{\text{denom}} = 0, \epsilon = 10^{-8}$）中，$h_{t, w}$ 仅成对出现在归一化与反归一化位置：
   $$\tilde{g}_{t, w} = M_{t-1, w} + h_{t, w}\,\psi_\alpha\!\left(\frac{g_{t, w} - M_{t-1, w}}{h_{t, w}}\right) = g_{t, w} - c_3 \frac{(g_{t, w} - M_{t-1, w})^3}{h_{t, w}^2} + O\!\left(\frac{|g_{t, w} - M_{t-1, w}|^5}{h_{t, w}^4}\right),$$
   即增大先验尺度 $h_{t, w}$ 以 $O(h_{t, w}^{-2})$ 阶减小对峰内干净梯度的非线性失真，同时分母 $\sigma_{t, w} = \sqrt{V_{t, w}} + 10^{-8}$ 不引入额外阻尼。 $\square$

---

### 定理 4（过峰回落 $\psi_\alpha$ 的冲击二阶矩渐近、偏差-方差权衡与对称单峰保号性）

固定历史 $\mathcal{F}_{t-1}$，设标准化新息 $z = \delta + \Xi$，其中真实标准化漂移 $\delta = \frac{\mu_{t, i} - M_{t-1, i}}{h_{t, i}}$，噪声 $\Xi \sim (1 - q)\mathcal{P}_{\text{clean}} + q\mathcal{P}_{\text{shock}}(\Lambda)$，$\mathcal{P}_{\text{shock}}(\Lambda) \overset{d}{=} \Lambda S$（$S$ 具有零对称、全局有界且在原点连续的概率密度 $p_S$，$p_S(0) > 0$）。

1. **（峰前逐位恒等）**：在事件 $\{|\delta + \Xi| \le z_*(\alpha)\}$ 上，$P_\alpha(\delta + \Xi) \equiv \psi_\alpha(\delta + \Xi)$。
2. **（冲击分支 $\Theta(\Lambda^{-1})$ 二阶矩消亡与有限漂移偏差权衡）**：对任意固定 $\delta \in \mathbb{R}$，当冲击尺度 $\Lambda \to \infty$ 时，由有界收敛与控制收敛定理：
   $$\lim_{\Lambda \to \infty} \mathbb{E}\!\left[P_\alpha(\delta + \Lambda S)^2\right] = \psi_{\max}^2, \qquad \lim_{\Lambda \to \infty} \Lambda \cdot \mathbb{E}\!\left[\psi_\alpha(\delta + \Lambda S)^2\right] = p_S(0)\|\psi_\alpha\|_{L^2(\mathbb{R})}^2 < \infty.$$
   由均方误差分解 $\operatorname{MSE}(\phi; \delta) = \operatorname{Var}(\phi) + (\mathbb{E}[\phi] - \delta)^2$：当 $|\delta|$ 小且 $\Lambda \gg 1$ 时，$\psi_\alpha$ 消除饱和方差平台 $\psi_{\max}^2$；但当真实漂移 $|\delta| > z_*$ 且冲击有限时，$\psi_\alpha$ 会产生大于单调核 $P_\alpha$ 的收缩偏差 $(\mathbb{E}[\psi_\alpha] - \delta)^2$。
3. **（对称单峰混合下的严格保号定理）**：若 $\mathcal{P}_{\text{clean}}$ 与 $S$ 均具有关于原点对称且在 $[0, \infty)$ 上单调不增的概率密度（涵盖高斯分布及一切对称稳定分布 $S\alpha_0 S, \; \alpha_0 \in (0, 2]$），则对任意 $q \in [0, 1], \Lambda > 0, \delta \ne 0$，令 $D(y) = p_\Xi(y - |\delta|) - p_\Xi(y + |\delta|) \ge 0$，由 $\int_0^\infty D(y)\,dy = \int_{-|\delta|}^{|\delta|} p_\Xi(u)\,du > 0$ 及 $\forall y > 0, \psi_\alpha(y) > 0$，恒有：
   $$\delta \cdot \mathbb{E}\!\left[\psi_\alpha(\delta + \Xi)\right] = |\delta| \int_0^\infty \psi_\alpha(y)\,D(y)\,dy > 0. \quad \square$$
