import math
from typing import List, Tuple, Union, Optional, Set
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

# 基础类型定义
# x y
Coordinates = tuple[float, float]
# 最小外接矩形：如下面所定义的坐标区间
# x_min y_min x_max y_max
MBR = tuple[float, float, float, float]

# 数据点定义：x,y为平面直角坐标，idx为点在某一表（如DataFrame，list[tuple]等等）当中的下标
class IndexedPoint:
    __slots__ = ("x", "y", "idx")
    def __init__(self, x: float, y: float, idx: int):
        self.x = x
        self.y = y
        self.idx = idx

    def __eq__(self, other):
        if isinstance(other, IndexedPoint):
            return self.x == other.x and self.y == other.y and self.idx == other.idx
        elif isinstance(other, tuple) and len(other) == 3:
            return (self.x, self.y, self.idx) == other
        return False

    def __hash__(self):
        return hash((self.x, self.y, self.idx))

    def __repr__(self):
        return f"({self.x}, {self.y}, idx={self.idx})"

    @property
    def point(self) -> Coordinates:
        return (self.x, self.y)

# R树的数据项（可以作为某一树的子树的数据）——点或具有最小外接矩形的结点
Entry = Union[IndexedPoint, "Node"]

# R树的节点
class Node:
    def __init__(self, is_leaf: bool = False):
        self.is_leaf = is_leaf
        self.entries: List[Entry] = []
        self.mbr: Optional[MBR] = None

    def update_mbr(self):
        """ 求所有子项的外接矩形作为MBR :return: N/A """
        if not self.entries:
            self.mbr = None
            return
        if self.is_leaf:
            xs = [p.x for p in self.entries]
            ys = [p.y for p in self.entries]
        else:
            xs = [child.mbr[0] for child in self.entries] + [
                child.mbr[2] for child in self.entries
            ]
            ys = [child.mbr[1] for child in self.entries] + [
                child.mbr[3] for child in self.entries
            ]
        self.mbr = (min(xs), min(ys), max(xs), max(ys))

# 计算MBR的面积
def mbr_area(mbr: MBR) -> float:
    return (mbr[2] - mbr[0]) * (mbr[3] - mbr[1])

# 从mbr1扩展到能够包括mbr2的范围
def expand_mbr(mbr1: Optional[MBR], mbr2: MBR) -> MBR:
    if mbr1 is None:
        return mbr2
    return (
        min(mbr1[0], mbr2[0]),
        min(mbr1[1], mbr2[1]),
        max(mbr1[2], mbr2[2]),
        max(mbr1[3], mbr2[3]),
    )

# 判断mbr1与mbr2是否相交
def intersects(mbr1: MBR, mbr2: MBR) -> bool:
    return not (
        mbr1[2] < mbr2[0]
        or mbr1[0] > mbr2[2]
        or mbr1[3] < mbr2[1]
        or mbr1[1] > mbr2[3]
    )

# 判断点是否在mbr当中
def point_in_mbr(p: IndexedPoint, mbr: MBR) -> bool:
    return mbr[0] <= p.x <= mbr[2] and mbr[1] <= p.y <= mbr[3]


class RTree:
    def __init__(self, max_entries: int = 4):
        self.max_entries = max_entries
        self.min_entries = max(1, math.ceil(max_entries / 2))
        self.root = Node(is_leaf=True)
        self._idx_counter = 0  # 在有需要的情况下用于累加获取idx

    def insert(self, x: float, y: float, idx: Optional[int] = None) -> int:
        """ 插入点，并返回实际使用的idx（如果调用函数时不输入有效的idx，则会通过计数器_idx_counter计算一个idx）。 注意idx参数的处理的一致性。 :param x: 点的x坐标，float类型 :param y: 点的y坐标，float类型 :param idx: 点在某一表（如DataFrame，list[tuple]等等）当中的下标，int类型 :return: 返回实际使用的idx """
        if idx is None:
            idx = self._idx_counter
            self._idx_counter += 1
        point = IndexedPoint(x, y, idx)
        leaf = self._choose_leaf(point)
        self._insert_in_leaf(leaf, point)
        if len(leaf.entries) > self.max_entries:
            self._split_node(leaf)
        return idx

    def _choose_leaf(self, point: IndexedPoint) -> Node:
        """ 选择将点插入哪个叶子结点：自根节点向下针对每一个子结点，计算如果插入新的结点MBR发生的面积增长，并以最小面积增长作为目标。 :param point: 当前正在判断的空间点数据 :return: 当前点应当插入的结点，Node类型 """
        node = self.root
        while not node.is_leaf:
            best_child = None
            best_area_increase = float("inf")
            best_area = float("inf")
            point_mbr = (point.x, point.y, point.x, point.y)
            for child in node.entries:
                new_mbr = expand_mbr(child.mbr, point_mbr)
                area_increase = mbr_area(new_mbr) - mbr_area(child.mbr)
                area = mbr_area(child.mbr)
                if area_increase < best_area_increase or (
                    area_increase == best_area_increase and area < best_area
                ):
                    best_area_increase = area_increase
                    best_area = area
                    best_child = child
            node = best_child
        return node

    def _insert_in_leaf(self, leaf: Node, point: IndexedPoint):
        """ 将结点插入叶子结点的过程，只应在insert方法当中调用 :param leaf: 插入到的叶子结点 :param point: 插入的空间点 :return: N/A """
        leaf.entries.append(point)
        leaf.update_mbr()

    def delete_by_idx(self, idx: int) -> bool:
        """ 仅仅通过 idx 删除点 :param idx: 将要删除的空间点的idx索引值 :return: 如果存在与输入索引值相一致的空间点并成功删除，则返回True， 如果未查找到与输入索引值相一致的空间点，则返回False。 """
        leaf = self._find_leaf_by_idx(self.root, idx)
        if leaf is None:
            return False
        # 找到具体点
        to_remove = None
        for p in leaf.entries:
            if p.idx == idx:
                to_remove = p
                break
        if to_remove is None:
            return False
        leaf.entries.remove(to_remove)
        leaf.update_mbr()
        self._condense_tree(leaf)
        if not self.root.is_leaf and len(self.root.entries) == 1:
            self.root = self.root.entries[0]
        return True

    def delete_point(self, x: float, y: float, idx: int) -> bool:
        """ 通过完整的空间信息与索引来删除点 :param x: 点的x坐标，float类型 :param y: 点的y坐标，float类型 :param idx: 空间点的索引值 :return: 如果存在与输入信息相一致的空间点并成功删除，则返回True， 如果未查找到与输入信息相一致的空间点，则返回False。 """
        point = IndexedPoint(x, y, idx)
        leaf = self._find_leaf_by_object(self.root, point)
        if leaf is None:
            return False
        if point in leaf.entries:
            leaf.entries.remove(point)
            leaf.update_mbr()
            self._condense_tree(leaf)
            if not self.root.is_leaf and len(self.root.entries) == 1:
                self.root = self.root.entries[0]
            return True
        return False

    def _find_leaf_by_idx(self, node: Node, target_idx: int) -> Optional[Node]:
        """ 在根为node的子树下找到索引值为target_idx的点所属的叶子结点。 :param node: 搜素的子树的根结点 :param target_idx: 搜索的点的idx索引值 :return: 如果查找到满足条件的叶子结点则返回对应的叶子结点，如果不存在满足条件的结点则返回None """
        if node.is_leaf:
            for p in node.entries:
                if p.idx == target_idx:
                    return node
            return None
        point_mbr_for_search = None  # 我们不知道位置，只能遍历所有可能子树
        for child in node.entries:
            # 启发式：只要 MBR 存在就递归（保守但正确）
            found = self._find_leaf_by_idx(child, target_idx)
            if found:
                return found
        return None

    def _find_leaf_by_object(self, node: Node, point: IndexedPoint) -> Optional[Node]:
        """ 通过空间点对象point（IndexedPoint）来找到点所属的叶子结点。 :param node: 子树的根结点 :param point: 查找的空间点对象，IndexedPoint对象 :return: 如果查找到满足条件的叶子结点则返回对应的叶子结点，如果不存在满足条件的结点则返回None """
        if node.is_leaf:
            return node if point in node.entries else None
        point_mbr = (point.x, point.y, point.x, point.y)
        for child in node.entries:
            if intersects(child.mbr, point_mbr):
                found = self._find_leaf_by_object(child, point)
                if found:
                    return found
        return None

    def query(self, region: MBR) -> list[IndexedPoint]:
        """ 查询一个矩形范围内的所有空间点（对象） :param region: 查询的矩形范围，格式与MBR相同 :return: 查询到的所有在矩形范围内的点的列表 """
        result = []
        self._query_recursive(self.root, region, result)
        return result

    def _query_recursive(self, node: Node, region: MBR, result: list[IndexedPoint]):
        """ 查询的递归方法 :param node: 查询的子树的根结点 :param region: 查询的矩形范围 :param result: 用于存储查询结果的列表 :return: N/A """
        if node.mbr is None or not intersects(node.mbr, region):
            return
        if node.is_leaf:
            for p in node.entries:
                if region[0] <= p.x <= region[2] and region[1] <= p.y <= region[3]:
                    result.append(p)
        else:
            for child in node.entries:
                self._query_recursive(child, region, result)

    def _find_parent(self, current: Node, target: Node) -> Optional[Node]:
        """ 从以current结点为根的子树当中找到结点target的父母结点 :param current: 子树的根结点 :param target: 目标结点 :return: 如果存在符合条件的结点则返回 """
        if current.is_leaf:
            return None
        for child in current.entries:
            if child is target:
                return current
        for child in current.entries:
            if not child.is_leaf:
                result = self._find_parent(child, target)
                if result:
                    return result
        return None

    def _condense_tree(self, node: Node):
        stack = []
        current = node
        parent = self._find_parent(self.root, current)
        while parent is not None:
            stack.append((parent, current))
            current = parent
            parent = self._find_parent(self.root, current)
        to_reinsert = []
        for parent, child in reversed(stack):
            if len(child.entries) < self.min_entries:
                parent.entries.remove(child)
                to_reinsert.extend(child.entries)
                parent.update_mbr()
            else:
                break
        for entry in to_reinsert:
            if isinstance(entry, IndexedPoint):
                self.insert(entry.x, entry.y, entry.idx)
            else:
                raise NotImplementedError("Only points supported")

    def _split_node(self, node: Node):
        group1, group2 = self._quadratic_split(node)
        parent = self._find_parent(self.root, node)
        if parent is None:
            new_root = Node(is_leaf=False)
            new_root.entries = [group1, group2]
            new_root.update_mbr()
            self.root = new_root
        else:
            parent.entries.remove(node)
            parent.entries.extend([group1, group2])
            parent.update_mbr()
            if len(parent.entries) > self.max_entries:
                self._split_node(parent)

    # ==================== 修复核心：使用索引而非 remove() ====================
    def _quadratic_split(self, node: Node):
        entries = node.entries[:]
        n = len(entries)

        # Step 1: Pick seeds by index
        worst_waste = -1
        seed_i, seed_j = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = entries[i], entries[j]
                mbr1 = (e1.x, e1.y, e1.x, e1.y) if node.is_leaf else e1.mbr
                mbr2 = (e2.x, e2.y, e2.x, e2.y) if node.is_leaf else e2.mbr
                combined = expand_mbr(mbr1, mbr2)
                waste = mbr_area(combined) - mbr_area(mbr1) - mbr_area(mbr2)
                if waste > worst_waste:
                    worst_waste = waste
                    seed_i, seed_j = i, j

        seed1 = entries[seed_i]
        seed2 = entries[seed_j]

        # Remove by index (safe!)
        if seed_i > seed_j:
            del entries[seed_i]
            del entries[seed_j]
        else:
            del entries[seed_j]
            del entries[seed_i]

        group1 = Node(is_leaf=node.is_leaf)
        group2 = Node(is_leaf=node.is_leaf)
        group1.entries.append(seed1)
        group2.entries.append(seed2)
        group1.update_mbr()
        group2.update_mbr()

        # Distribute remaining entries
        while entries:
            # Check if one group must take all remaining
            if len(group1.entries) + len(entries) == self.min_entries:
                group1.entries.extend(entries)
                break
            if len(group2.entries) + len(entries) == self.min_entries:
                group2.entries.extend(entries)
                break

            entry = entries.pop()
            entry_mbr = (entry.x, entry.y, entry.x, entry.y) if node.is_leaf else entry.mbr

            mbr1_new = expand_mbr(group1.mbr, entry_mbr)
            mbr2_new = expand_mbr(group2.mbr, entry_mbr)
            d1 = mbr_area(mbr1_new) - mbr_area(group1.mbr)
            d2 = mbr_area(mbr2_new) - mbr_area(group2.mbr)

            if d1 < d2:
                group1.entries.append(entry)
                group1.mbr = mbr1_new
            elif d2 < d1:
                group2.entries.append(entry)
                group2.mbr = mbr2_new
            else:
                if len(group1.entries) <= len(group2.entries):
                    group1.entries.append(entry)
                    group1.mbr = mbr1_new
                else:
                    group2.entries.append(entry)
                    group2.mbr = mbr2_new

        group1.update_mbr()
        group2.update_mbr()
        return group1, group2

    # ======================================================================

    # ================== 可视化（可以带 idx 标注）==================
    def visualize(self, ax=None, show_points=True, show_mbrs=True, annotate_idx=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 9))
            ax.set_aspect("equal")
            created_fig = True
        else:
            created_fig = False

        # 获取所有点
        all_points = self.query(
            (-float("inf"), -float("inf"), float("inf"), float("inf"))
        )
        if show_points and all_points:
            xs = [p.x for p in all_points]
            ys = [p.y for p in all_points]
            ax.scatter(xs, ys, c="black", s=1, zorder=10, label="Points")
            if annotate_idx:
                for p in all_points:
                    ax.text(
                        p.x + 0.1,
                        p.y + 0.1,
                        str(p.idx),
                        fontsize=9,
                        color="darkblue",
                    )

        # 绘制分层 MBR
        if show_mbrs:
            max_depth = self._get_max_depth(self.root)
            cmap = plt.get_cmap("tab10")
            self._draw_mbrs_by_level(
                self.root, ax, level=0, max_depth=max_depth, cmap=cmap
            )

        if created_fig:
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.set_title("R-Tree with Indexed Points", fontsize=14)
            plt.tight_layout()
            plt.savefig("RTree.png")
            # plt.show()

    def _get_max_depth(self, node: Node, depth: int = 0) -> int:
        if node.is_leaf or not node.entries:
            return depth
        return max(self._get_max_depth(child, depth + 1) for child in node.entries)

    def _draw_mbrs_by_level(self, node: Node, ax, level: int, max_depth: int, cmap):
        if node.mbr is None:
            return
        min_x, min_y, max_x, max_y = node.mbr
        width = max_x - min_x
        height = max_y - min_y
        color = cmap(level % cmap.N)
        rect = patches.Rectangle(
            (min_x, min_y),
            width,
            height,
            linewidth=0.5,
            edgecolor=color,
            facecolor="none",
            linestyle="-",
            label=f"Level {level}" if level == 0 else "",
        )
        ax.add_patch(rect)
        if not node.is_leaf:
            for child in node.entries:
                self._draw_mbrs_by_level(child, ax, level + 1, max_depth, cmap)


def to_dict(node):
    if node is None:
        return None
    dict_node = {"MBR": node.mbr, "entries": []}
    for entry in node.entries:
        if isinstance(entry, Node):  # 如果是子节点，则递归处理
            dict_entry = to_dict(entry)
        else:  # 假定entry是一个点(元组)
            dict_entry = entry
        dict_node["entries"].append(dict_entry)
    return dict_node


def from_dict(dict_node, parent=None):
    node = Node(parent=parent)
    if dict_node is not None:
        node.mbr = dict_node.get("MBR")
        for dict_entry in dict_node.get("entries", []):
            if isinstance(dict_entry, dict):  # 如果是子节点字典
                child_node = from_dict(dict_entry, parent=node)
                node.entries.append(child_node)
            else:  # 假定dict_entry是一个点(元组)
                node.entries.append(dict_entry)
    return node


# =============== 示例用法 (AI生成) ===============
if __name__ == "__main__":
    import random

    # ====== 配置参数 ======
    NUM_POINTS = 30  # ←←← 在这里设置点的数量
    MAX_ENTRIES = 4  # R 树节点最大容量
    DELETE_COUNT = 5  # 随机删除几个点
    RANDOM_SEED = 42  # 固定随机种子，确保可复现
    random.seed(RANDOM_SEED)

    # 生成不重复的随机点（在 [0, 20) × [0, 20) 范围内）
    points_set = set()
    while len(points_set) < NUM_POINTS:
        x = round(random.uniform(0, 20), 1)
        y = round(random.uniform(0, 20), 1)
        points_set.add((x, y))
    points_list = list(points_set)[:NUM_POINTS]  # 确保数量精确
    print(f"✅ 生成 {len(points_list)} 个随机点")

    # ====== 构建 R 树 ======
    rtree = RTree(max_entries=MAX_ENTRIES)
    # 插入所有点，idx = 列表索引
    for idx, (x, y) in enumerate(points_list):
        rtree.insert(x, y, idx=idx)
    print(
        f"✅ 插入完成，总点数: {len(rtree.query((-float('inf'), -float('inf'), float('inf'), float('inf'))))}"
    )

    # ====== 范围查询示例 ======
    query_region = (5, 5, 15, 15)
    results = rtree.query(query_region)
    print(f"🔍 在区域 {query_region} 中找到 {len(results)} 个点")
    for p in results[:5]:  # 只打印前5个
        print(f" {p}")
    if len(results) > 5:
        print(f" ... 还有 {len(results) - 5} 个点")

    # ====== 随机删除一些点 ======
    all_idxs = [
        p.idx
        for p in rtree.query(
            (-float("inf"), -float("inf"), float("inf"), float("inf"))
        )
    ]
    to_delete = random.sample(all_idxs, min(DELETE_COUNT, len(all_idxs)))
    print(f"🗑️ 尝试删除 {len(to_delete)} 个点，idx: {sorted(to_delete)}")
    deleted_count = 0
    for idx in to_delete:
        if rtree.delete_by_idx(idx):
            deleted_count += 1
    remaining = len(
        rtree.query((-float("inf"), -float("inf"), float("inf"), float("inf")))
    )
    print(f"✅ 成功删除 {deleted_count} 个点，剩余 {remaining} 个点")

    # ====== 可视化 ======
    print("🖼️ 正在绘制 R 树结构（带 idx 标注）...")
    rtree.visualize(annotate_idx=True)