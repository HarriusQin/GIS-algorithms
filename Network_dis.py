from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import heapq
from scipy.spatial import cKDTree
from collections import defaultdict

# 图数据类型：邻接表
NID = int                     # 节点ID
Weight = float                # 边权重（现在为距离，单位：米）
Coord = tuple[float, float]   # 点的xy坐标
Edge = tuple[NID, NID, Weight]  # 边 (u, v, weight)
Graph = dict[NID, list[tuple[NID, Weight]]]  # 邻接表


def extract_nodes_and_edges(
    gdf_roads: gpd.GeoDataFrame,
    speed_map: Optional[dict[int, float]] = None  # 保留参数以保持接口兼容，但不再使用
) -> tuple[list[Coord], list[Edge], dict[Coord, NID]]:
    """
    从路网 GeoDataFrame 提取节点、边和坐标到 ID 的映射。
    边的权重为几何长度（单位：米），即按距离计算。

    :param gdf_roads: 路网数据，必须包含 'length' 和几何列（LineString），geopandas.GeoDataFrame类型
    :param speed_map: 保留参数以保持接口兼容，但本函数中不再使用
    :return: (node_coords, edges, coord_to_id)
    """
    node_coords: list[Coord] = []
    coord_to_id: dict[Coord, NID] = {}
    edges: list[Edge] = []

    node_id: NID = 0

    for _, row in gdf_roads.iterrows():
        geom = row.geometry
        if geom.geom_type != 'LineString':
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        start: Coord = coords[0]
        end: Coord = coords[-1]

        for pt in (start, end):
            if pt not in coord_to_id:
                coord_to_id[pt] = node_id
                node_coords.append(pt)
                node_id += 1

        u = coord_to_id[start]
        v = coord_to_id[end]
        weight = float(row['length'])  # 直接使用长度作为距离权重（单位：米）

        edges.append((u, v, weight))
        edges.append((v, u, weight))  # 双向图

    return node_coords, edges, coord_to_id


def extract_nodes_and_edges_by_distance(
    gdf_roads: gpd.GeoDataFrame,
    speed_map: Optional[dict[int, float]] = None  # 保留参数以保持接口兼容
) -> tuple[list[Coord], list[Edge], dict[Coord, NID]]:
    """
    从路网 GeoDataFrame 提取节点、边和坐标到 ID 的映射。
    边的权重为几何长度（单位：米），即按距离计算。
    （此函数与 extract_nodes_and_edges 功能一致，保留用于兼容或明确语义）

    :param gdf_roads: 路网数据，必须包含 'length' 和几何列（LineString）
    :param speed_map: 保留参数以保持接口兼容，但本函数中不再使用
    :return: (node_coords, edges, coord_to_id)
    """
    return extract_nodes_and_edges(gdf_roads, speed_map)


def build_graph(edges: list[Edge]) -> Graph:
    """从边列表构建邻接表图"""
    graph: Graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
    return graph


def dijkstra_pathfinding(
    graph: Graph,
    start: NID,
    goal: NID
) -> tuple[float, list[NID], int]:
    """
    使用 Dijkstra 算法寻找最短距离路径。
    返回 (最短距离（米）, 路径节点列表, 访问节点数)
    若不可达，返回 (inf, [], visited_count)
    """
    dist: dict[NID, float] = defaultdict(lambda: float('inf'))
    prev: dict[NID, NID] = {}
    dist[start] = 0.0
    pq: list[tuple[float, NID]] = [(0.0, start)]
    visited_count: int = 0

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        visited_count += 1
        if u == goal:
            break
        for v, w in graph[u]:
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    # Reconstruct path
    path: list[NID] = []
    curr: NID = goal
    if curr != start and curr not in prev:
        return float('inf'), [], visited_count

    while curr in prev:
        path.append(curr)
        curr = prev[curr]
    path.append(start)
    path.reverse()

    return dist[goal], path, visited_count


def a_star_pathfinding(
    graph: Graph,
    start: NID,
    goal: NID,
    node_coords: list[Coord],
    avg_speed_mps: float = 12.0  # 保留参数但不再用于距离计算；启发函数改用欧氏距离
) -> tuple[float, list[NID], int]:
    """
    A* 算法（距离最优），使用欧氏距离作为启发函数（单位：米）。
    注意：avg_speed_mps 参数已无实际作用，仅为保留函数签名。
    """
    def heuristic(n1: NID, n2: NID) -> float:
        x1, y1 = node_coords[n1]
        x2, y2 = node_coords[n2]
        dx = x2 - x1
        dy = y2 - y1
        euclid = (dx * dx + dy * dy) ** 0.5
        manhattan_distance = abs(dx) + abs(dy)
        return euclid  # 欧氏距离（米）

    open_set: list[tuple[float, NID]] = []
    heapq.heappush(open_set, (heuristic(start, goal), start))

    g_score: dict[NID, float] = defaultdict(lambda: float('inf'))
    g_score[start] = 0.0

    f_score: dict[NID, float] = defaultdict(lambda: float('inf'))
    f_score[start] = heuristic(start, goal)

    came_from: dict[NID, NID] = {}
    visited_count: int = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_count += 1

        if current == goal:
            path: list[NID] = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return g_score[goal], path, visited_count

        for neighbor, dist_cost in graph[current]:
            tentative_g = g_score[current] + dist_cost
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return float('inf'), [], visited_count


def snap_point_to_network(point: Point, node_coords: list[Coord]) -> NID:
    """将 POI 点匹配到最近的路网节点（基于欧氏距离）"""
    tree = cKDTree(node_coords)
    _, idx = tree.query([point.x, point.y])
    return int(idx)


def visualize_network_and_path(
    gdf_roads: gpd.GeoDataFrame,
    gdf_poi: gpd.GeoDataFrame,
    start_row: gpd.GeoSeries,
    end_row: gpd.GeoSeries,
    result_gdf: gpd.GeoDataFrame,
    figsize: tuple[int, int] = (12, 10),
    output_filename: str = "shortest_path_by_distance.png"
) -> None:
    """
    使用 GeoPandas 内置绘图功能可视化最短路径（按距离）：
    - 路网：浅灰色细线（背景）
    - 所有 POI：小橙点（半透明，弱化显示）
    - 起点：绿色大圆圈（带黑边）
    - 终点：红色大三角（带黑边）
    - 最短路径：蓝色粗线（若起点终点相同则为紫色点）

    :param gdf_roads: 路网数据（已投影，单位：米）
    :param gdf_poi: POI 数据（与路网同 CRS）
    :param start_row: 起点 POI 行（GeoSeries）
    :param end_row: 终点 POI 行（GeoSeries）
    :param result_gdf: 最短路径结果（LineString 或 Point）
    :param figsize: 图像尺寸
    :param output_filename: 输出图像文件名
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 1. 绘制路网（背景）
    gdf_roads.plot(ax=ax, color='lightgray', linewidth=0.6, zorder=1)

    # 2. 绘制所有 POI（弱化显示，避免喧宾夺主）
    if not gdf_poi.empty:
        gdf_poi.plot(ax=ax, color='orange', markersize=8, alpha=0.5, zorder=2, label='Other POIs')

    # 3. 绘制最短路径
    if not result_gdf.empty:
        geom_type = result_gdf.geometry.iloc[0].geom_type
        if geom_type == 'LineString':
            result_gdf.plot(ax=ax, color='blue', linewidth=3, zorder=4, label='Shortest Path (by Distance)')
        elif geom_type == 'Point':
            result_gdf.plot(ax=ax, color='purple', marker='o', markersize=100, zorder=4, label='Same Start/End Node')

    # 4. 单独高亮起点和终点（zorder 确保在最上层）
    gpd.GeoDataFrame([start_row]).plot(
        ax=ax, color='green', marker='o', markersize=150, edgecolor='black', linewidth=1.2, zorder=5, label='Start'
    )
    gpd.GeoDataFrame([end_row]).plot(
        ax=ax, color='red', marker='^', markersize=150, edgecolor='black', linewidth=1.2, zorder=5, label='End'
    )

    # 5. 美化
    ax.set_title("Shortest Path by Distance", fontsize=16, fontweight='bold')
    ax.set_axis_off()
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    # 6. 自动缩放到路径或起终点范围
    if not result_gdf.empty and result_gdf.geometry.iloc[0].geom_type == 'LineString':
        bounds = result_gdf.total_bounds
    else:
        # 如果路径退化为点，使用起点+终点的包围盒
        all_points = gpd.GeoDataFrame([start_row, end_row])
        bounds = all_points.total_bounds

    margin_x = max((bounds[2] - bounds[0]) * 0.1, 100)  # 至少留100米边距
    margin_y = max((bounds[3] - bounds[1]) * 0.1, 100)
    ax.set_xlim(bounds[0] - margin_x, bounds[2] + margin_x)
    ax.set_ylim(bounds[1] - margin_y, bounds[3] + margin_y)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"可视化图像已保存为: {output_filename}")
