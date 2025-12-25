from __future__ import annotations

import importlib.util
import os
import sys

import networkx as nx


# ---------- 动态加载 SZlSolver ----------

def load_szl_solver_class() -> type:
    here = os.path.dirname(os.path.abspath(__file__))
    solver_path = os.path.join(here, "SZl-identify.py")
    spec = importlib.util.spec_from_file_location("szl_identify_module_test_sz3", solver_path)
    if spec is None or spec.loader is None:
        raise ImportError("无法加载 SZl-identify.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "SZlSolver"):
        raise ImportError("SZl-identify.py 中未找到 SZlSolver")
    return module.SZlSolver  # type: ignore[return-value]


# ---------- 构建图 ----------

def build_graph_1() -> nx.MultiGraph:
    """图1：7个点，边集为12,13,17,23,24,26,35,37,45,46,67,57（每条边1重）"""
    Gm = nx.MultiGraph()
    Gm.add_nodes_from([1, 2, 3, 4, 5,6,7,8,9,10,11,12])
    edges = [
         (1, 9), (1, 10),(1,11),(1,12),
         (2, 7), (2,10),(2,11),(2,12),
         (3, 7),(3,8),(3,11),(3,12),
         (4, 7),(4,8),(4,9),(4,12),
         (5,7),(5,8),(5,9),(5,10),
         (6,8),(6,9),(6,10),(6,11)
    ]
    for u, v in edges:
        Gm.add_edge(u, v)

    return Gm


# ---------- 主流程 ----------

def main():
    print("=" * 60)
    print("图1：7个点，边集为12,13,17,23,24,26,35,37,45,46,67,57")
    print("=" * 60)

    Gm = build_graph_1()

    # 输出图信息
    print(f"\n图信息：")
    print(f"- 顶点数: {len(Gm.nodes())}")
    print(f"- 顶点列表: {sorted(Gm.nodes())}")

    # 统计边
    edges_list = list(Gm.edges())
    print(f"- 边数: {len(edges_list)}")
    print(f"- 边列表: {sorted([(min(u, v), max(u, v)) for u, v in edges_list])}")

    # 计算度数
    deg = dict(Gm.degree())
    print(f"- 顶点度: {deg}")

    # 检查连通性
    if not nx.is_connected(nx.Graph(Gm)):
        print("\n警告：该图不连通，SZlSolver 要求图必须连通！")
        print("连通分量：")
        for comp in nx.connected_components(nx.Graph(Gm)):
            print(f"  {sorted(comp)}")
        return

    # SZ3 判定
    print(f"\n开始 SZ3 判定（l=3）...")
    try:
        SZlSolver = load_szl_solver_class()
        solver = SZlSolver(Gm, 3)

        is_sz3, witness_beta = solver.is_SZl(verbose=False)
    except Exception as e:
        print(f"\n错误：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n结果：")
    if is_sz3:
        print(f"该图是 SZ_3 ✓")
    else:
        print(f"该图不是 SZ_3 ✗")
        if witness_beta is not None:
            vertices = sorted(solver.vertices)
            vec = [witness_beta[v] for v in vertices]
            print(f"\n反例 beta（向量表示）：{solver.format_beta(witness_beta)}")

            # 尝试求解该beta，看看是否能得到一些信息
            ok, assign = solver.solve_for_beta(witness_beta)
            if not ok:
                print("该 beta 在 SZ3 下确实无解")
            else:
                print("注意：该 beta 在 SZ3 下有解，但程序判定为不可行，可能存在bug")


if __name__ == "__main__":
    main()