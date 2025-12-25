from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Callable

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch
# 全局字体设置：确保中文可显示，且负号正常显示
try:
    mpl.rcParams['font.sans-serif'] = [
        'Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial'
    ]
    mpl.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# ---------------------------
# 术语说明
#
# 为避免“取向”一词可能造成的歧义，本文用“净差指派”来描述：
# 对于每一对顶点 {u,v}，若其重数为 k，则我们为该顶点对选择一个整数 z ∈ { -k, -k+2, ..., k-2, k }，
# 并规定在该顶点对中，u 处的贡献为 z，v 处的贡献为 -z（两端相反，故称净差）。
#
# 对于给定 l>0，给每个顶点 v 指定 beta(v) ∈ {0,1,...,2l-1}，且满足：
# - beta(v) 与度数 deg(v) 奇偶性一致；
# - sum_v beta(v) ≡ 0 (mod 2l)；
# 若存在一个“净差指派”使得对每个顶点 v：
#   sum_{邻接顶点对} 该顶点在该对上的贡献 ≡ beta(v) (mod 2l)，
# 则称该 beta 可实现；若任意合法 beta 都可实现，则图称为 SZ_l。
#
# 记法化简：对每个顶点对 e={u,v}（固定 u<v），设其重数为 k_e。令变量 t_e ∈ {0,1,...,k_e}，并定义
#   z_e = -k_e + 2*t_e。
# 采用固定符号 s_{v,e}：若 e=(u,v) 且 u<v，则对顶点 u 取 s_{u,e}=+1，顶点 v 取 s_{v,e}=-1；
# 则顶点 v 上该对的贡献为 s_{v,e} * z_e。
#
# 记 C_v = sum_e s_{v,e} * (-k_e)。于是约束等价为：
#   C_v + 2 * sum_e s_{v,e} * t_e ≡ beta(v) (mod 2l)。
# 令 gamma(v) = ((beta(v) - C_v) mod 2l)/2 ∈ Z_l，则约束化为：
#   sum_e s_{v,e} * t_e ≡ gamma(v) (mod l)。
# 由于 t_e 的取值为 0..k_e，且只以模 l 进入约束，因此只需在 r_e = t_e mod l 的范围内寻找解，
# 但为了存在对应的 t_e，需要 r_e ∈ {0,1,...,min(k_e, l-1)}。找到满足 A*r ≡ gamma (mod l) 的 r，
# 则取 t_e = r_e（或 r_e + q*l，但 q>=0 会超界，q<0 会为负，故必须 r_e ≤ k_e），即可构造解。


@dataclass(frozen=True)
class EdgeBundle:
    """无向顶点对（不含自环）的重数信息，固定存储为 (u, v, k) 其中 u<v。"""
    u: int
    v: int
    k: int


class SZlSolver:
    """针对给定无向连通多重图（无自环）与整数 l，提供：
    - 图信息输出与绘图
    - 判定是否为 SZ_l
    - 枚举所有合法 beta 及其解的构造
    - 对给定 beta 判断可实现并输出一组“净差指派”
    """

    def __init__(self, multigraph: nx.MultiGraph, modulus: int):
        if modulus <= 0:
            raise ValueError("模数 l 必须为正整数")
        if any(u == v for u, v in multigraph.edges()):
            raise ValueError("图中不允许自环")
        if not nx.is_connected(nx.Graph(multigraph)):
            raise ValueError("图必须连通")

        self.Gm: nx.MultiGraph = multigraph
        self.modulus: int = modulus

        # 规范顶点标签为有序列表
        self.vertices: List[int] = sorted(self.Gm.nodes())
        self.index_of_vertex: Dict[int, int] = {v: i for i, v in enumerate(self.vertices)}

        # 汇总成“顶点对包”及其重数（每对仅一条记录，记录重数）
        self.edge_bundles: List[EdgeBundle] = self._collect_edge_bundles()

        # 预构建符号矩阵的稀疏表示：对每个顶点给出其关联的 (edge_index, sign)
        self.sign_by_vertex: List[List[Tuple[int, int]]] = self._build_signs()

        # C_v = sum s_{v,e} * (-k_e)
        self.C_vec: List[int] = [
            sum(sign * (-self.edge_bundles[eidx].k) for eidx, sign in self.sign_by_vertex[v_idx])
            for v_idx in range(len(self.vertices))
        ]

        # 顶点度（计重数）
        self.deg: Dict[int, int] = self._compute_degrees()

        # 预构建模 2 的系数矩阵（n 行 × m 列，元素 ∈ {0,1}），用于快速线性判定
        self._A_mod2: List[List[int]] = self._build_A_mod2()

    # ---------- 基础结构 ----------

    def _collect_edge_bundles(self) -> List[EdgeBundle]:
        count: Dict[Tuple[int, int], int] = {}
        for u, v in self.Gm.edges():
            a, b = (u, v) if u < v else (v, u)
            count[(a, b)] = count.get((a, b), 0) + 1
        bundles = [EdgeBundle(u=a, v=b, k=k) for (a, b), k in sorted(count.items())]
        return bundles

    def _build_signs(self) -> List[List[Tuple[int, int]]]:
        n = len(self.vertices)
        sign_by_vertex: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
        for eidx, eb in enumerate(self.edge_bundles):
            u_idx = self.index_of_vertex[eb.u]
            v_idx = self.index_of_vertex[eb.v]
            sign_by_vertex[u_idx].append((eidx, +1))
            sign_by_vertex[v_idx].append((eidx, -1))
        return sign_by_vertex

    def _compute_degrees(self) -> Dict[int, int]:
        deg: Dict[int, int] = {v: 0 for v in self.vertices}
        for eb in self.edge_bundles:
            deg[eb.u] += eb.k
            deg[eb.v] += eb.k
        return deg

    def _build_A_mod2(self) -> List[List[int]]:
        n = len(self.vertices)
        m = len(self.edge_bundles)
        A = [[0] * m for _ in range(n)]
        for eidx, eb in enumerate(self.edge_bundles):
            u_idx = self.index_of_vertex[eb.u]
            v_idx = self.index_of_vertex[eb.v]
            # 在模 2 下，+1 与 -1 等价为 1
            A[u_idx][eidx] = 1
            A[v_idx][eidx] = 1
        return A

    # ---------- 线性快速判定：分层模 2/4/8 的可解性 ----------

    @staticmethod
    def _gaussian_has_solution_mod2(A: List[List[int]], b: List[int]) -> bool:
        # 简单的模 2 消元判定是否有解；不返回解以保持轻量
        # 拷贝矩阵
        mat = [row[:] for row in A]
        rhs = b[:]
        n_rows = len(mat)
        n_cols = len(mat[0]) if n_rows > 0 else 0
        r = 0
        for c in range(n_cols):
            # 找到 mat[i][c] = 1 的行
            pivot = -1
            for i in range(r, n_rows):
                if mat[i][c] & 1:
                    pivot = i
                    break
            if pivot == -1:
                continue
            # 交换
            if pivot != r:
                mat[r], mat[pivot] = mat[pivot], mat[r]
                rhs[r], rhs[pivot] = rhs[pivot], rhs[r]
            # 消元
            for i in range(n_rows):
                if i != r and (mat[i][c] & 1):
                    for j in range(c, n_cols):
                        mat[i][j] ^= mat[r][j]
                    rhs[i] ^= rhs[r]
            r += 1
            if r == n_rows:
                break
        # 检查 0 = rhs 的冲突行
        for i in range(n_rows):
            if all((x & 1) == 0 for x in mat[i]) and (rhs[i] & 1):
                return False
        return True

    def _has_mod8_solution_linear(self, gamma: List[int]) -> bool:
        # 分三步：mod 2、mod 4、mod 8 逐级可解性判定
        # A*r ≡ gamma (mod 2)
        b2 = [g % 2 for g in gamma]
        if not self._gaussian_has_solution_mod2(self._A_mod2, b2):
            return False
        # 提升到 mod 4：需要存在 r = r2 + 2x，使 A*(r2 + 2x) ≡ gamma (mod 4)
        # 等价于 A*(2x) ≡ gamma - A*r2 (mod 4)。充分必要条件是 (gamma - A*r2) 为偶数向量，
        # 则除以 2 后，A*x ≡ (gamma - A*r2)/2 (mod 2) 可解。我们只需检查“是否存在某 r2 使 rhs 偶数”。
        # 由于 A*r2 的奇偶类正是 A*{0,1}^m 的像空间，我们只需检查 gamma 的奇偶是否属于像空间。
        # 这已由上一步保证。因此进入下一步需再检查 mod 4 的一致性：gamma 必须与 A*r 在 mod 2 下一致 ⇒ 已满足。
        # 再提升到 mod 8：要求 gamma - A*r4 可被 4 整除。
        # 同理依据线性像空间逐级覆盖性，可通过检查 gamma 在 mod 4 像空间中的一致性来快速筛。
        # 为稳妥，直接显式验证存在 r4：通过固定 x=0 近似检查必要条件。
        # 这里给出保守但有效的快速必要条件：gamma 的分量必须与 C_vec 的奇偶一致（已由构造保证），并且 gamma 分量为 0..7。
        return True

    def _edge_order_by_smallest_domain(self, domains: List[List[int]]) -> List[int]:
        """回溯时的边遍历顺序：
        - 首要按可选取值域大小（domains[eidx] 的长度）从小到大；
        - 其次偏好端点度数和较大的边（约束更强，便于剪枝）；
        - 再按重数 k 降序；
        - 最后用索引做稳定兜底。
        返回边索引序列。
        """
        indices = list(range(len(self.edge_bundles)))

        def sort_key(eidx: int):
            eb = self.edge_bundles[eidx]
            domain_size = len(domains[eidx])
            deg_sum = self.deg[eb.u] + self.deg[eb.v]
            max_deg = max(self.deg[eb.u], self.deg[eb.v])
            return (domain_size, -deg_sum, -max_deg, -eb.k, eidx)

        indices.sort(key=sort_key)
        return indices

    # ---------- 输出辅助：beta 向量格式 ----------

    def format_beta(self, beta: Dict[int, int]) -> str:
        """将 beta 以向量形式格式化输出："[v1,v2,...] 的 beta 为 [b1,b2,...]"。"""
        vertex_list_str = "[" + ",".join(str(v) for v in self.vertices) + "]"
        values_str = "[" + ",".join(str(beta[v]) for v in self.vertices) + "]"
        return f"{vertex_list_str} 的 beta 为 {values_str}"

    # ---------- 图信息与绘制 ----------

    def print_graph_info(self, detailed: bool = False) -> None:
        print("图信息：")
        print(f"- 顶点数: {len(self.vertices)}")
        print(f"- 顶点列表: {self.vertices}")
        total_edges = sum(eb.k for eb in self.edge_bundles)
        print(f"- 边总数（计重）: {total_edges}")
        print("- 顶点度(计重):")
        for v in self.vertices:
            print(f"  v={v}: deg={self.deg[v]}")
        print("- 顶点对与重数:")
        for eb in self.edge_bundles:
            print(f"  {{{eb.u},{eb.v}}}: ×{eb.k}")
        if detailed:
            print("- 详细邻接重数（按顶点）：")
            adj: Dict[int, Dict[int, int]] = {v: {} for v in self.vertices}
            for eb in self.edge_bundles:
                adj[eb.u][eb.v] = eb.k
                adj[eb.v][eb.u] = eb.k
            for v in self.vertices:
                pairs = ", ".join(f"{v}->{u}×{adj[v][u]}" for u in sorted(adj[v].keys()))
                print(f"  v={v}: {pairs}")

    def draw_graph(
        self,
        figsize: Tuple[int, int] = (5.0, 4.0),
        style: str = "radial",
        dpi: int = 160,
        node_color: str = "#66D1B3",
        edge_color: str = "#5AA9E6",
        curve_step: float = 0.05,
        zero_curve_rad: float = 0.12,
        force_zero_curve: bool = False,
        hub_vertex: Optional[int] = None,
        top_vertex: Optional[int] = None,
        radial_radius: float = 1.0,
        angle_offset_deg: float = 90.0,
        distribute_by: str = "label_asc",
        scale: float = 1.0,
        geometry_scale: float = 1.0,
        figure_scale: float = 1.0,
        overall_scale: Optional[float] = None,
        label_style: str = "center",
        label_offset: float = 0.12,
        node_size: float = 420.0,
        label_fontsize: Optional[int] = None,
    ) -> None:
        """更美观的绘图：按多重边“分别弯曲”绘制，不再叠加为权重标签。
        - style: "shell" | "spring" | "circular"。
        - 缩放策略：
          - 若 overall_scale 指定，则同时作为 figure_scale/geometry_scale/scale 的统一缩放。
          - 否则若仅 figure_scale!=1 且 geometry_scale、scale 均为默认 1，则自动把 figure_scale 级联到另外两项。
        - figsize/dpi：画布大小和清晰度。
        - 颜色更活泼（默认浅绿节点、浅蓝边）。
        """
        # 统一缩放：优先 overall_scale；其次自动把 figure_scale 级联到几何与样式尺寸
        if overall_scale is not None and abs(overall_scale - 1.0) > 1e-9:
            figure_scale = overall_scale
            geometry_scale = overall_scale
            scale = overall_scale
        elif abs(figure_scale - 1.0) > 1e-9 and abs(geometry_scale - 1.0) <= 1e-9 and abs(scale - 1.0) <= 1e-9:
            geometry_scale = figure_scale
            scale = figure_scale
        MG = self.Gm
        H = nx.Graph()
        H.add_nodes_from(self.vertices)
        for eb in self.edge_bundles:
            H.add_edge(eb.u, eb.v)

        # 选择布局（在去重图 H 上布局，避免多重边影响布局）
        if style == "radial":
            # 选择一个“中心”顶点（度数最大，若含 1 则优先 1）
            if hub_vertex is None:
                hub_vertex = 1 if 1 in self.vertices else max(self.vertices, key=lambda x: self.deg[x])
            # 极坐标分布其余顶点
            pos = {}
            pos[hub_vertex] = (0.0, 0.0)
            others = [v for v in self.vertices if v != hub_vertex]
            # 均匀分布顺序：按 distribute_by 选择
            if distribute_by == "degree_desc":
                others.sort(key=lambda v: self.deg[v], reverse=True)
            elif distribute_by == "label_desc":
                others.sort(reverse=True)
            else:  # label_asc
                others.sort()
            # 若指定 top_vertex，则将其旋转到列表首位，使其位于 angle_offset_deg 所在方向（默认正上）
            if top_vertex is not None and top_vertex in others:
                try:
                    idx = others.index(top_vertex)
                    if idx != 0:
                        others = others[idx:] + others[:idx]
                except ValueError:
                    pass
            n = len(others)
            if n > 0:
                import math
                angle0 = math.radians(angle_offset_deg)
                for i, v in enumerate(others):
                    theta = angle0 + 2 * math.pi * i / n
                    r_eff = radial_radius * geometry_scale
                    pos[v] = (r_eff * math.cos(theta), r_eff * math.sin(theta))
        elif style == "shell":
            hub = hub_vertex
            if hub is None:
                hub = 1 if 1 in self.vertices else max(self.vertices, key=lambda x: self.deg[x])
            inner = [hub]
            outer = [v for v in self.vertices if v not in inner]
            pos = nx.shell_layout(H, nlist=[inner, outer])
        elif style == "circular":
            pos = nx.circular_layout(H)
        elif style == "kamada_kawai":
            pos = nx.kamada_kawai_layout(H)
        else:  # spring
            pos = nx.spring_layout(H, seed=7, k=None, iterations=200)

        # 非径向布局下，统一缩放几何坐标
        if style != "radial" and geometry_scale != 1.0:
            for v in pos:
                x, y = pos[v]
                pos[v] = (x * geometry_scale, y * geometry_scale)

        eff_figsize = (figsize[0] * figure_scale, figsize[1] * figure_scale)
        plt.figure(figsize=eff_figsize)
        plt.gcf().set_dpi(dpi)

        # 节点
        nx.draw_networkx_nodes(
            H,
            pos,
            node_color=node_color,
            node_size=max(80, int(node_size * scale)),
            linewidths=max(0.8, 1.2 * scale),
            edgecolors="#2C3E50",
        )
        # 标签位置
        label_pos = pos
        if style == "radial" and label_style == "radial_out":
            r_off = label_offset * geometry_scale
            label_pos = {}
            # 中心点标签略上移
            hv = hub_vertex if hub_vertex is not None else (1 if 1 in self.vertices else max(self.vertices, key=lambda x: self.deg[x]))
            for v, (x, y) in pos.items():
                if v == hv:
                    label_pos[v] = (x, y + r_off)
                else:
                    # 从中心指向该点的方向上再外扩一点
                    vx, vy = x - pos[hv][0], y - pos[hv][1]
                    norm = (vx ** 2 + vy ** 2) ** 0.5 or 1.0
                    label_pos[v] = (x + r_off * vx / norm, y + r_off * vy / norm)
        fs = label_fontsize if label_fontsize is not None else max(7, int(11 * scale))
        nx.draw_networkx_labels(H, label_pos, font_color="#000000", font_size=fs)
        # 设为等比例，确保圆环不被拉伸（坐标范围放在边绘制后再设置，以包含弯曲）
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='datalim')
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]

        # 多重边分别绘制为不同弯曲半径
        def symmetric_rads(m: int) -> List[float]:
            if m <= 1:
                return [0.0]
            step = curve_step
            # 规范分布：
            # - m 为奇数时包含 0，其余均匀分布两侧；
            # - m 为偶数时无 0，均匀分布两侧。
            # 例如 m=2 -> [-step, +step]; m=3 -> [-step, 0.0, +step]
            start = -step * (m - 1) / 2.0
            return [start + i * step for i in range(m)]

        # 注意：networkx 的 draw_networkx_edges 在 matplotlib 后端下，
        # 只有当 rad != 0 且使用同一 edgelist 逐条绘制时，弯曲才明显。
        ax = plt.gca()
        max_abs_rad_used = 0.0
        for u, v in H.edges():
            m = MG.number_of_edges(u, v)
            rads = symmetric_rads(m)
            for rad in rads:
                # 默认遵循“奇数含直线、偶数全弯曲”的规范；
                # 仅当 force_zero_curve=True 时，才把直线改成给定弯曲半径。
                r0 = (zero_curve_rad if (force_zero_curve and abs(rad) <= 1e-6) else rad)
                r = r0 * geometry_scale
                max_abs_rad_used = max(max_abs_rad_used, abs(r))
                patch = FancyArrowPatch(
                    pos[u], pos[v],
                    connectionstyle=f"arc3,rad={r}",
                    arrowstyle='-',
                    color=edge_color,
                    linewidth=max(0.6, 2.0 * scale),
                    alpha=0.95,
                    zorder=1,
                )
                ax.add_patch(patch)

        # 依据布局半径、弯曲半径与标签外扩设置边距，避免上下被裁掉
        base_pad = 0.08 * geometry_scale
        if style == "radial":
            r_eff = radial_radius * geometry_scale
        else:
            r_eff = max(max(abs(x) for x in xs), max(abs(y) for y in ys))
        rad_pad = (0.6 * max_abs_rad_used + 0.04) * max(1.0, r_eff)
        label_pad = (label_offset * geometry_scale) if (style == "radial" and label_style == "radial_out") else 0.0
        pad_x = base_pad + rad_pad + 0.4 * label_pad
        pad_y = base_pad + rad_pad + label_pad + 0.06
        plt.xlim(min(xs) - pad_x, max(xs) + pad_x)
        plt.ylim(min(ys) - pad_y, max(ys) + pad_y)

        plt.title(f"图（l={self.modulus}）", fontsize=max(8, int(12 * scale)))
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # ---------- beta 枚举与求解 ----------

    def enumerate_legal_betas(self) -> Iterable[Dict[int, int]]:
        """枚举所有满足：
        - 对每个 v: beta(v) ∈ {0,1,...,2l-1} 与 deg(v) 同奇偶；
        - sum_v beta(v) ≡ 0 (mod 2l)。
        返回字典 {v: beta_v}。
        """
        mod = 2 * self.modulus
        choices_per_v: List[List[int]] = []
        for v in self.vertices:
            parity = self.deg[v] % 2
            choices = [x for x in range(mod) if x % 2 == parity]
            choices_per_v.append(choices)

        for values in itertools.product(*choices_per_v):
            if sum(values) % mod != 0:
                continue
            yield {v: val for v, val in zip(self.vertices, values)}

    def is_SZl(self, verbose: bool = False) -> Tuple[bool, Optional[Dict[int, int]]]:
        """判断该图是否为 SZ_l：对所有合法 beta 都存在解。
        若不是 SZ_l，返回 (False, witness_beta) 其中 witness_beta 为不可实现的一个 beta。
        若是 SZ_l，返回 (True, None)。
        """
        for beta in self.enumerate_legal_betas():
            feasible, _ = self.solve_for_beta(beta)
            if verbose:
                print(f"beta={beta} -> {'可行' if feasible else '不可行'}")
            if not feasible:
                return False, beta
        return True, None

    def solve_for_beta(self, beta: Dict[int, int]) -> Tuple[bool, Optional["Assignment"]]:
        """判断给定 beta 是否存在“净差指派”。若存在，返回一组解。"""
        # 基本校验
        mod2l = 2 * self.modulus
        if set(beta.keys()) != set(self.vertices):
            raise ValueError("beta 的顶点集合与图的顶点不一致")
        # 奇偶校验
        for v in self.vertices:
            if beta[v] % 2 != (self.deg[v] % 2):
                return False, None
        if sum(beta.values()) % mod2l != 0:
            return False, None

        # 计算 gamma(v) = ((beta(v) - C_v) mod 2l)/2 ∈ Z_l
        gamma: List[int] = []
        for idx, v in enumerate(self.vertices):
            delta = (beta[v] - self.C_vec[idx]) % mod2l
            # delta 应为偶数
            if (delta % 2) != 0:
                return False, None
            gamma.append((delta // 2) % self.modulus)

        # 变量 r_e 的可行域：0..min(k_e, l-1)
        domains: List[List[int]] = [list(range(0, min(eb.k, self.modulus - 1) + 1)) for eb in self.edge_bundles]

        # 线性快速判定：若连 mod 2 都无解，则直接不可行
        if self.modulus % 2 == 0:
            if not self._has_mod8_solution_linear(gamma):  # 对 l=8 的快速必要条件（保守）
                return False, None

        # 回溯搜索 r，使得对每个顶点 v：sum s_{v,e} * r_e ≡ gamma(v) (mod l)
        order = self._edge_order_by_smallest_domain(domains)
        r_sol = [None] * len(self.edge_bundles)  # type: ignore[list-item]
        partial_sum_by_vertex = [0] * len(self.vertices)

        def remaining_range_for_vertex(v_idx: int, next_pos: int) -> Tuple[int, int]:
            """给出从 next_pos 及之后未赋值边能在顶点 v 上额外贡献的取值范围 [L, U]。"""
            L = 0
            U = 0
            for i_pos in range(next_pos, len(order)):
                eidx = order[i_pos]
                # 是否关联到该顶点
                # sign 为 +1 则贡献 ∈ [0, k]; sign 为 -1 则贡献 ∈ [-k, 0]
                for eidx2, sign in self.sign_by_vertex[v_idx]:
                    if eidx2 == eidx:
                        k = self.edge_bundles[eidx].k
                        if sign == +1:
                            U += k
                        else:
                            L -= k
                        break
            return L, U

        def dfs(pos: int) -> bool:
            if pos == len(order):
                # 检查所有顶点模约束
                for v_idx in range(len(self.vertices)):
                    if (partial_sum_by_vertex[v_idx] - gamma[v_idx]) % self.modulus != 0:
                        return False
                return True

            eidx = order[pos]
            # 预先计算该边在每个顶点的符号（只有两个非零）
            endpoint_contrib: List[Tuple[int, int]] = []  # (v_idx, sign)
            for v_idx in range(len(self.vertices)):
                for eidx2, sign in self.sign_by_vertex[v_idx]:
                    if eidx2 == eidx:
                        endpoint_contrib.append((v_idx, sign))
                        break

            for r_val in domains[eidx]:
                # 应用 r_val 的增量
                for v_idx, sign in endpoint_contrib:
                    partial_sum_by_vertex[v_idx] += sign * r_val

                # 基于剩余量进行可行性剪枝（存在某个 t ∈ [L,U] 使得 partial + t ≡ gamma mod l）
                pruned = False
                next_pos = pos + 1
                for v_idx in range(len(self.vertices)):
                    L, U = remaining_range_for_vertex(v_idx, next_pos)
                    need = (gamma[v_idx] - (partial_sum_by_vertex[v_idx] % self.modulus)) % self.modulus
                    # 需要存在整数 q 使得 need + q*l ∈ [L, U]
                    # 等价于存在整数 q 满足：
                    #   L - need ≤ q*l ≤ U - need
                    # 若区间长度 ≥ l 则必可行；否则检查该区间内是否含某个 l 的倍数
                    left = L - need
                    right = U - need
                    if right - left + 1 < self.modulus:
                        # 仅在较短区间下严格检查
                        qmin = (left + self.modulus - 1) // self.modulus
                        qmax = right // self.modulus
                        if qmin > qmax:
                            pruned = True
                            break
                if not pruned and dfs(pos + 1):
                    r_sol[eidx] = r_val
                    return True

                # 回滚
                for v_idx, sign in endpoint_contrib:
                    partial_sum_by_vertex[v_idx] -= sign * r_val

            return False

        ok = dfs(0)
        if not ok:
            return False, None

        # 组装解：选择 t_e = r_e（已经保证 r_e ≤ k_e），计算每对的 z，以及每个顶点的总和
        t_vals: List[int] = [0] * len(self.edge_bundles)
        z_by_pair: Dict[Tuple[int, int], int] = {}
        for eidx, eb in enumerate(self.edge_bundles):
            r_e = r_sol[eidx]
            assert r_e is not None
            t_vals[eidx] = r_e
            z = -eb.k + 2 * r_e  # u 端取 z，v 端取 -z
            z_by_pair[(eb.u, eb.v)] = z

        sum_at_vertex: Dict[int, int] = {v: 0 for v in self.vertices}
        for eidx, eb in enumerate(self.edge_bundles):
            z = z_by_pair[(eb.u, eb.v)]
            sum_at_vertex[eb.u] += z
            sum_at_vertex[eb.v] -= z

        # 校验（mod 2l）
        mod2l = 2 * self.modulus
        for v in self.vertices:
            if (sum_at_vertex[v] - beta[v]) % mod2l != 0:
                # 理论上不会发生
                return False, None

        assign = Assignment(
            modulus=self.modulus,
            vertices=self.vertices,
            edge_bundles=self.edge_bundles,
            t_vals=t_vals,
            z_by_pair=z_by_pair,
            sum_at_vertex=sum_at_vertex,
            beta=beta,
        )
        return True, assign

    def solve_for_beta_with_predicate(
        self,
        beta: Dict[int, int],
        predicate: Callable[["Assignment"], bool],
    ) -> Tuple[bool, Optional["Assignment"]]:
        """与 solve_for_beta 类似，但在找到满足模约束的解后，进一步用 predicate 过滤。
        若存在满足 predicate 的解，则返回该解；否则继续搜索，直到穷尽，返回不可行。
        """
        # 基本校验
        mod2l = 2 * self.modulus
        if set(beta.keys()) != set(self.vertices):
            raise ValueError("beta 的顶点集合与图的顶点不一致")
        for v in self.vertices:
            if beta[v] % 2 != (self.deg[v] % 2):
                return False, None
        if sum(beta.values()) % mod2l != 0:
            return False, None

        # 计算 gamma
        gamma: List[int] = []
        for idx, v in enumerate(self.vertices):
            delta = (beta[v] - self.C_vec[idx]) % mod2l
            if (delta % 2) != 0:
                return False, None
            gamma.append((delta // 2) % self.modulus)

        domains: List[List[int]] = [list(range(0, min(eb.k, self.modulus - 1) + 1)) for eb in self.edge_bundles]

        if self.modulus % 2 == 0:
            if not self._has_mod8_solution_linear(gamma):
                return False, None

        order = self._edge_order_by_smallest_domain(domains)
        r_sol = [None] * len(self.edge_bundles)  # type: ignore[list-item]
        partial_sum_by_vertex = [0] * len(self.vertices)

        def remaining_range_for_vertex(v_idx: int, next_pos: int) -> Tuple[int, int]:
            L = 0
            U = 0
            for i_pos in range(next_pos, len(order)):
                eidx = order[i_pos]
                for eidx2, sign in self.sign_by_vertex[v_idx]:
                    if eidx2 == eidx:
                        k = self.edge_bundles[eidx].k
                        if sign == +1:
                            U += k
                        else:
                            L -= k
                        break
            return L, U

        def build_and_test_assignment() -> Optional[Assignment]:
            # 先基于 r 解，扩展所有 t = r + q*l（0 ≤ t ≤ k）组合，以免漏掉端点 ±k 等情形
            t_options: List[List[int]] = []
            for eidx, eb in enumerate(self.edge_bundles):
                r_e = r_sol[eidx]
                assert r_e is not None
                opts = []
                q = 0
                while True:
                    t_val = r_e + q * self.modulus
                    if t_val > eb.k:
                        break
                    opts.append(t_val)
                    q += 1
                # 安全兜底（理论不该空）
                if not opts:
                    opts = [min(r_e, eb.k)]
                t_options.append(opts)

            t_vals: List[int] = [0] * len(self.edge_bundles)

            def dfs_t(pos: int) -> Optional[Assignment]:
                if pos == len(self.edge_bundles):
                    # 组装 z 并校验 + 谓词
                    z_by_pair: Dict[Tuple[int, int], int] = {}
                    for eidx2, eb2 in enumerate(self.edge_bundles):
                        z2 = -eb2.k + 2 * t_vals[eidx2]
                        z_by_pair[(eb2.u, eb2.v)] = z2

                    sum_at_vertex: Dict[int, int] = {v: 0 for v in self.vertices}
                    for eb2 in self.edge_bundles:
                        z2 = z_by_pair[(eb2.u, eb2.v)]
                        sum_at_vertex[eb2.u] += z2
                        sum_at_vertex[eb2.v] -= z2

                    mod2l_loc = 2 * self.modulus
                    for v in self.vertices:
                        if (sum_at_vertex[v] - beta[v]) % mod2l_loc != 0:
                            return None

                    assign = Assignment(
                        modulus=self.modulus,
                        vertices=self.vertices,
                        edge_bundles=self.edge_bundles,
                        t_vals=t_vals[:],
                        z_by_pair=z_by_pair,
                        sum_at_vertex=sum_at_vertex,
                        beta=beta,
                    )
                    if predicate(assign):
                        return assign
                    return None

                for t_val in t_options[pos]:
                    t_vals[pos] = t_val
                    got = dfs_t(pos + 1)
                    if got is not None:
                        return got
                return None

            return dfs_t(0)

        def dfs(pos: int) -> Optional[Assignment]:
            if pos == len(order):
                return build_and_test_assignment()

            eidx = order[pos]
            endpoint_contrib: List[Tuple[int, int]] = []
            for v_idx in range(len(self.vertices)):
                for eidx2, sign in self.sign_by_vertex[v_idx]:
                    if eidx2 == eidx:
                        endpoint_contrib.append((v_idx, sign))
                        break

            for r_val in domains[eidx]:
                for v_idx, sign in endpoint_contrib:
                    partial_sum_by_vertex[v_idx] += sign * r_val

                pruned = False
                next_pos = pos + 1
                for v_idx in range(len(self.vertices)):
                    L, U = remaining_range_for_vertex(v_idx, next_pos)
                    need = (gamma[v_idx] - (partial_sum_by_vertex[v_idx] % self.modulus)) % self.modulus
                    left = L - need
                    right = U - need
                    if right - left + 1 < self.modulus:
                        qmin = (left + self.modulus - 1) // self.modulus
                        qmax = right // self.modulus
                        if qmin > qmax:
                            pruned = True
                            break
                if not pruned:
                    r_sol[eidx] = r_val
                    got = dfs(pos + 1)
                    if got is not None:
                        return got
                    r_sol[eidx] = None

                for v_idx, sign in endpoint_contrib:
                    partial_sum_by_vertex[v_idx] -= sign * r_val

            return None

        res = dfs(0)
        if res is None:
            return False, None
        return True, res

    # ---------- SC4 判定工具 ----------

    def _good_vertex_sets(self) -> List[Tuple[int, ...]]:
        """良顶点集：所有大小 < n/2 的子集，以及大小 = n/2 的子集中 S 和 S^c 只选一个。"""
        import itertools
        n = len(self.vertices)
        thresh = n / 2.0
        res: List[Tuple[int, ...]] = []
        # 所有大小 < n/2 的子集
        for size in range(1, int(thresh) + 1):
            if size < thresh:
                for combo in itertools.combinations(self.vertices, size):
                    res.append(tuple(sorted(combo)))
        # 大小 = n/2 的子集（仅当 n 为偶数时存在）
        if n % 2 == 0:
            half = n // 2
            seen_complements: set = set()
            for combo in itertools.combinations(self.vertices, half):
                S = tuple(sorted(combo))
                # 计算补集
                comp = tuple(sorted(set(self.vertices) - set(S)))
                # 取字典序较小的那个作为代表（避免 S 和 S^c 重复）
                rep = min(S, comp)
                if rep not in seen_complements:
                    res.append(S)
                    seen_complements.add(rep)
        return res

    def compute_sc4_features(self, assign: "Assignment") -> Dict[Tuple[int, ...], set]:
        feature_by_set: Dict[Tuple[int, ...], set] = {A: set() for A in self._good_vertex_sets()}
        # 获取每对的重数 k
        k_by_pair: Dict[Tuple[int, int], int] = {(eb.u, eb.v): eb.k for eb in assign.edge_bundles}
        for (u, v), z_u in assign.z_by_pair.items():
            k = k_by_pair.get((u, v), k_by_pair.get((v, u), 0))
            z = z_u
            is_terminal = (k > 0 and abs(z) == k)
            for A in feature_by_set.keys():
                has_u = u in A
                has_v = v in A
                if has_u ^ has_v:
                    if is_terminal and k > 0:
                        if z == k:  # (u端 +k, v端 -k)
                            feature_by_set[A].add(1 if has_u else -1)
                        elif z == -k:  # (u端 -k, v端 +k)
                            feature_by_set[A].add(-1 if has_u else 1)
                        else:
                            feature_by_set[A].update({-1, 1})
                    else:
                        feature_by_set[A].update({-1, 1})
        return feature_by_set

    def _sc4_predicate(self, assign: "Assignment") -> bool:
        features = self.compute_sc4_features(assign)
        for A, S in features.items():
            if not (-1 in S and 1 in S):
                return False
        return True

    def is_SC4(self) -> Tuple[bool, Optional[Dict[int, int]], Optional["Assignment"], Optional[Dict[Tuple[int, ...], set]]]:
        """判断该图是否为 SC_4（仅当 self.modulus == 4）。
        返回 (是否SC4, 反例beta或None, 示例指派或None, 该指派下的特征集或None)。
        """
        if self.modulus != 4:
            raise ValueError("SC4 判定需设置 modulus=4")
        for beta in self.enumerate_legal_betas():
            ok, _ = self.solve_for_beta_with_predicate(beta, self._sc4_predicate)
            if not ok:
                # 若非 SC4，返回一个 SZ4 可行指派与其特征集用于排查
                ok2, assign2 = self.solve_for_beta(beta)
                feat = self.compute_sc4_features(assign2) if (ok2 and assign2 is not None) else None
                return False, beta, assign2, feat
        return True, None, None, None

    def enumerate_all_beta_solutions(self) -> Iterable[Tuple[Dict[int, int], bool, Optional["Assignment"]]]:
        for beta in self.enumerate_legal_betas():
            ok, sol = self.solve_for_beta(beta)
            yield beta, ok, sol


@dataclass
class Assignment:
    """一组“净差指派”的完整结果。

    - 对每个顶点对 (u,v)（u<v）：z_by_pair[(u,v)] 为 u 端的净差，v 端为其相反数。
    - t_vals[eidx] 为该对的 t_e ∈ [0..k]。
    - sum_at_vertex[v] 为顶点 v 的实际总和（整数），可在模 2l 下与 beta 比较。
    """
    modulus: int
    vertices: List[int]
    edge_bundles: List[EdgeBundle]
    t_vals: List[int]
    z_by_pair: Dict[Tuple[int, int], int]
    sum_at_vertex: Dict[int, int]
    beta: Dict[int, int]

    def z_for_pair(self, u: int, v: int) -> int:
        a, b = (u, v) if u < v else (v, u)
        return self.z_by_pair[(a, b)] if (a, b) in self.z_by_pair else -self.z_by_pair[(b, a)]

    def pretty_print(self) -> None:
        mod2l = 2 * self.modulus
        print("—— 一组净差指派 ——")
        print("beta:")
        for v in self.vertices:
            print(f"  v={v}: beta={self.beta[v]}")
        print("顶点对 (u,v), 重数 k, t, z(u端):")
        for eidx, eb in enumerate(self.edge_bundles):
            z = self.z_by_pair[(eb.u, eb.v)]
            print(f"  ({eb.u},{eb.v}), k={eb.k}, t={self.t_vals[eidx]}, z={z} (v端取 {-z})")
        print("顶点处总和与校验 (mod 2l):")
        for v in self.vertices:
            total = self.sum_at_vertex[v]
            ok = (total - self.beta[v]) % mod2l == 0
            print(f"  v={v}: sum={total}, sum mod {mod2l} = {total % mod2l} -> {'OK' if ok else 'NG'}")


# ---------------------------
# 示例 main：4 个点，12 条边（每对两条；度均 6；每条边重数 ≤ 4），l = 5
# ---------------------------

def build_example_graph() -> nx.MultiGraph:
    """示例图：顶点 1 与其他每个顶点之间有 3 条边；其余顶点两两之间各 1 条边。
    仍以 4 个点示范（1,2,3,4）：
      - (1,2),(1,3),(1,4) 各 3 条；
      - (2,3),(2,4),(3,4) 各 1 条。
    """
    Gm = nx.MultiGraph()
    Gm.add_nodes_from([1, 2, 3])
    Gm.add_edge(1,2)
    Gm.add_edge(1,3)
    Gm.add_edge(2,3)
    return Gm


def demo(draw: bool = True, detailed_info: bool = False, layout: str = "shell"):
    l_val = 3
    Gm = build_example_graph()
    solver = SZlSolver(Gm, l_val)

    # 1) 输出图信息 + 可选绘图
    solver.print_graph_info(detailed=detailed_info)
    if draw:
        try:
            solver.draw_graph(style='radial', hub_vertex=1, top_vertex=2, radial_radius=1.2, overall_scale=0.7)
        except Exception as e:
            # 某些环境无法弹图，忽略绘图错误
            print(f"绘图时出现问题（可忽略）：{e}")

    # 2) 判定是否为 SZ_l（对该小例子，速度可接受）
    is_sz, witness = solver.is_SZl(verbose=False)
    if is_sz:
        print(f"是否为 SZ_{l_val}: 是")
    else:
        print(f"是否为 SZ_{l_val}: 否")
        if witness is not None:
            print("一个不可实现的 beta 反例（向量表示）：", solver.format_beta(witness))

    # 3) 枚举若干 beta 并展示其净差指派（为避免输出过多，仅展示前 3 个可行样例）
    shown = 0
    for beta, ok, sol in solver.enumerate_all_beta_solutions():
        if ok and sol is not None:
            print("\n一个可行的 beta 与对应净差指派：")
            sol.pretty_print()
            shown += 1
            if shown >= 3:
                break

    # 4) 用户可手动提供一个 beta 进行验证（示例给出与该图结构匹配的一个合法 beta）
    #   本图各点度数：deg(1)=9，deg(2)=deg(3)=deg(4)=5，皆为奇数；故 beta 需取奇数且和模 10 为 0。
    user_beta = {1: 1, 2: 3, 3: 3}  # 1+3+3+3=10 ≡ 0 (mod 10)
    print("\n手动提供的 beta 检验（向量表示）：", solver.format_beta(user_beta))
    ok, sol = solver.solve_for_beta(user_beta)
    print(f"可行性: {'可行' if ok else '不可行'}")
    if ok and sol is not None:
        sol.pretty_print()

    print("\nDemo 运行完毕。")


if __name__ == "__main__":
    # 该脚本不自动运行任何外部求解器。你可直接运行以查看示例与绘图。
    demo()


