from __future__ import annotations

import importlib.util
import os
import sys

import networkx as nx

print("当前运行的文件路径：", os.path.abspath(__file__))
# ---------- 动态加载 SZlSolver ----------

def load_szl_solver_class() -> type:
    here = os.path.dirname(os.path.abspath(__file__))
    solver_path = os.path.join(here, "SZl-identify.py")
    spec = importlib.util.spec_from_file_location("szl_identify_module_test_sz3", solver_path)
    if spec is None or spec.loader is None:
        raise ImportError("fail to load SZl-identify.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "SZlSolver"):
        raise ImportError("SZl-identify.py does not contain SZlSolver")
    return module.SZlSolver  # type: ignore[return-value]


# ---------- 构建图 ----------



#bigraphic pair (4^3,3^2;4^3,3^2)
def build_graph_1() -> nx.MultiGraph:
    G_1 = nx.MultiGraph()
    G_1.add_nodes_from([1, 2, 3, 4, 5,6,7,8,9,10])
    edges = [
         (1, 7), (1, 8),(1,9),(1,10),
         (2, 6), (2,8),(2,9),(2,10),
         (3, 6),(3,7),(3,9),(3,10),
         (4, 6),(4,7),(4,10),
         (5,6),(5,7),(5,8)
    ]
    for u, v in edges:
        G_1.add_edge(u, v)
    return G_1

#bigraphic pair (4^3,3^2;5,4,3^3)
def build_graph_2() -> nx.MultiGraph:
    G_2 = nx.MultiGraph()
    G_2.add_nodes_from([1, 2, 3, 4, 5,6,7,8,9,10])
    edges = [
         (1, 6), (1, 7),(1,8),(1,9),
         (2, 6), (2,8),(2,9),(2,10),
         (3, 6),(3,7),(3,9),(3,10),
         (4, 6),(4,7),(4,10),
         (5,6),(5,7),(5,8)
    ]
    for u, v in edges:
        G_2.add_edge(u, v)
    return G_2

#bigraphic pair (5,4,3^3;5,4,3^3)
def build_graph_3() -> nx.MultiGraph:
    G_3= nx.MultiGraph()
    G_3.add_nodes_from([1, 2, 3, 4, 5,6,7,8,9,10])
    edges = [
        (1, 6), (1, 7), (1, 8), (1, 9),(1,10),
        (2, 6), (2, 7), (2, 8), (2, 9),
        (3, 6), (3, 8), (3, 9),
        (4, 6), (4, 7), (4, 10),
        (5, 6), (5, 7), (5, 10)
    ]
    for u, v in edges:
        G_3.add_edge(u, v)
    return G_3

#bigraphic pair (4^6;4^6)
def build_graph_4() -> nx.MultiGraph:
    G_4 = nx.MultiGraph()
    G_4.add_nodes_from([1, 2, 3, 4, 5,6,7,8,9,10,11,12])
    edges = [
         (1, 9), (1, 10),(1,11),(1,12),
         (2, 7), (2,10),(2,11),(2,12),
         (3, 7),(3,8),(3,11),(3,12),
         (4, 7),(4,8),(4,9),(4,12),
         (5,7),(5,8),(5,9),(5,10),
         (6,8),(6,9),(6,10),(6,11)
    ]
    for u, v in edges:
        G_4.add_edge(u, v)
    return G_4

#bigraphic pair (4^7;4^7)
def build_graph_5() -> nx.MultiGraph:
    G_5 = nx.MultiGraph()
    G_5.add_nodes_from([1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14])
    edges = [
         (1, 9), (1, 10),(1,11),(1,12),
         (2, 10), (2,11),(2,12),(2,13),
         (3, 11),(3,12),(3,13),(3,14),
         (4, 8),(4,12),(4,13),(4,14),
         (5,8),(5,9),(5,13),(5,14),
         (6,8),(6,9),(6,10),(6,14),
         (7,8),(7,9),(7,10),(7,11)
    ]
    for u, v in edges:
        G_5.add_edge(u, v)
    return G_5


# ---------- 主流程 ----------

def main():
    G_1 = build_graph_1()
    G_2 = build_graph_2()
    G_3 = build_graph_3()
    G_4 = build_graph_4()
    G_5 = build_graph_5()

    # 统计边
    edges_list_1 = list(G_1.edges())
    edges_list_2 = list(G_2.edges())
    edges_list_3 = list(G_3.edges())
    edges_list_4 = list(G_4.edges())
    edges_list_5 = list(G_5.edges())


    # 计算度数
    deg_1 = dict(G_1.degree())
    deg_2 = dict(G_2.degree())
    deg_3 = dict(G_3.degree())
    deg_4 = dict(G_4.degree())
    deg_5 = dict(G_5.degree())

    # 检查连通性
    if not nx.is_connected(nx.Graph(G_1)):
        print("\n警告：该图不连通，SZlSolver 要求图必须连通！")
        print("连通分量：")
        for comp in nx.connected_components(nx.Graph(G_1)):
            print(f"  {sorted(comp)}")
        return
    if not nx.is_connected(nx.Graph(G_2)):
        print("\n警告：该图不连通，SZlSolver 要求图必须连通！")
        print("连通分量：")
        for comp in nx.connected_components(nx.Graph(G_2)):
            print(f"  {sorted(comp)}")
        return
    if not nx.is_connected(nx.Graph(G_3)):
        print("\n警告：该图不连通，SZlSolver 要求图必须连通！")
        print("连通分量：")
        for comp in nx.connected_components(nx.Graph(G_3)):
            print(f"  {sorted(comp)}")
        return
    if not nx.is_connected(nx.Graph(G_4)):
        print("\n警告：该图不连通，SZlSolver 要求图必须连通！")
        print("连通分量：")
        for comp in nx.connected_components(nx.Graph(G_4)):
            print(f"  {sorted(comp)}")
        return
    if not nx.is_connected(nx.Graph(G_5)):
        print("\n警告：该图不连通，SZlSolver 要求图必须连通！")
        print("连通分量：")
        for comp in nx.connected_components(nx.Graph(G_5)):
            print(f"  {sorted(comp)}")
        return

    # Z3 判定

    try:
        SZlSolver = load_szl_solver_class()
        solver = SZlSolver(G_1, 3)

        is_sz3, witness_beta = solver.is_SZl(verbose=False)
    except Exception as e:
        print(f"\nerror：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nresult：")
    if is_sz3:
        print(f"G_1 is Z_3 connected ✓")
    else:
        print(f"G_1 is not  Z_3 connected ✗")

    try:
        SZlSolver = load_szl_solver_class()
        solver = SZlSolver(G_2, 3)

        is_sz3, witness_beta = solver.is_SZl(verbose=False)
    except Exception as e:
        print(f"\nerror：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nresult：")
    if is_sz3:
        print(f"G_2 is Z_3 connected ✓")
    else:
        print(f"G_2 is not  Z_3 connected ✗")

    try:
        SZlSolver = load_szl_solver_class()
        solver = SZlSolver(G_3, 3)

        is_sz3, witness_beta = solver.is_SZl(verbose=False)
    except Exception as e:
        print(f"\nerror：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nresult：")
    if is_sz3:
        print(f"G_3 is Z_3 connected ✓")
    else:
        print(f"G_3 is not  Z_3 connected ✗")

    try:
        SZlSolver = load_szl_solver_class()
        solver = SZlSolver(G_4, 3)

        is_sz3, witness_beta = solver.is_SZl(verbose=False)
    except Exception as e:
        print(f"\nerror：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nresult：")
    if is_sz3:
        print(f"G_4 is Z_3 connected ✓")
    else:
        print(f"G_4 is not  Z_3 connected ✗")

    try:
        SZlSolver = load_szl_solver_class()
        solver = SZlSolver(G_5, 3)

        is_sz3, witness_beta = solver.is_SZl(verbose=False)
    except Exception as e:
        print(f"\nerror：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\nresult：")
    if is_sz3:
        print(f"G_5 is Z_3 connected ✓")
    else:
        print(f"G_5 is not  Z_3 connected ✗")


if __name__ == "__main__":
    main()